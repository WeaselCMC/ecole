#include <fmt/format.h>
#include <map>

#include <xtensor/xarray.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "ecole/instance/ship-allocation.hpp"
#include "ecole/scip/cons.hpp"
#include "ecole/scip/model.hpp"
#include "ecole/scip/utils.hpp"
#include "ecole/scip/var.hpp"

namespace ecole::instance {

/*************************************
 *  SAPGenerator methods  *
 *************************************/

SAPGenerator::SAPGenerator(Parameters parameters_, RandomGenerator rng_) : rng{rng_}, parameters{parameters_} {}

SAPGenerator::SAPGenerator(Parameters parameters_) : SAPGenerator{parameters_, ecole::spawn_random_generator()} {}

SAPGenerator::SAPGenerator() : SAPGenerator(Parameters{}) {}

scip::Model SAPGenerator::next() {
	return generate_instance(parameters, rng);
}

void SAPGenerator::seed(Seed seed) {
	rng.seed(seed);
}

namespace {

using std::size_t;
using xvector = xt::xtensor<size_t, 1>;

/** Adds a variable to the SCIP Model.
 *
 * Adds a single binary variable with a specified cost.
 */
auto add_var(SCIP* scip, size_t i, size_t j, size_t k, SCIP_Real cost) -> SCIP_VAR* {
	auto const name = fmt::format("x_{}_{}_{}", i, j, k);
	auto unique_var = scip::create_var_basic(scip, name.c_str(), 0., 1., cost, SCIP_VARTYPE_BINARY);
	auto* var_ptr = unique_var.get();
	scip::call(SCIPaddVar, scip, var_ptr);
	return var_ptr;
}

/** Adds all variables to the SCIP Model.
 *
 */
auto add_vars(SCIP* scip, xt::xtensor<SCIP_Real, 3> const& c) -> xt::xtensor<SCIP_VAR*, 1> {
	auto shape = c.shape();
	auto vars = xt::xtensor<SCIP_VAR*, 3>{{shape}};

	for (size_t i = 0; i < shape[0]; ++i) {
		for (size_t j = 0; j < shape[1]; ++j) {
			for (size_t k = 0; k < shape[2]; ++k) {
				vars(i, j, k) = add_var(scip, i, j, k, c(i, j, k));
			}
		}
	}

	return vars;
}

/*
 *
 */
auto add_constaints(
	SCIP* scip,
	xt::xtensor<SCIP_VAR*, 3> vars,
	xt::xtensor<SCIP_Real, 3> max_prod,
	xt::xtensor<SCIP_Real, 2> month_constr,
	xt::xtensor<SCIP_Real, 1> annual_constr) {

	auto const neg_inf = -SCIPinfinity(scip);

	for (size_t j = 0; j < vars.shape(1); ++j) {
		for (size_t i = 0; i < vars.shape(0); ++i) {
			for (size_t k = 0; k < vars.shape(2); ++k) {
				auto name = fmt::format("Y_{}_{}_{}", i, j, k);
				auto coefs = xt::xtensor<SCIP_Real, 1>({1}, 1.);
				auto cons = scip::create_cons_basic_linear(
					scip, name.c_str(), 1, &vars(i, j, k), coefs.data(), neg_inf, max_prod(i, j, k));
				scip::call(SCIPaddCons, scip, cons.get());
			}

			xt::xtensor<SCIP_VAR*, 1> x_ij = xt::view(vars, i, j, xt::all());
			auto name = fmt::format("E_{}_{}", i, j);
			auto coefs = xt::xtensor<SCIP_Real, 1>({vars.shape(2)}, 1.);
			auto cons = scip::create_cons_basic_linear(
				scip, name.c_str(), x_ij.size(), &x_ij(0), coefs.data(), neg_inf, month_constr(i, j));
			scip::call(SCIPaddCons, scip, cons.get());
		}

		xt::xtensor<SCIP_VAR*, 1> x_j = xt::flatten(xt::view(vars, xt::all(), j, xt::all()));
		auto name = fmt::format("Z_{}", j);
		auto coefs = xt::xtensor<SCIP_Real, 1>({vars.shape(0) * vars.shape(2)}, 1.);
		auto cons =
			scip::create_cons_basic_linear(scip, name.c_str(), x_j.size(), &x_j(0), coefs.data(), neg_inf, annual_constr(j));
		scip::call(SCIPaddCons, scip, cons.get());
	}
}

}  // namespace

/******************************************
 *  SAPGenerator::generate_instance  *
 ******************************************/

scip::Model SAPGenerator::generate_instance(Parameters parameters, RandomGenerator& rng) {
	auto const n_months = parameters.n_months;
	auto const n_places = parameters.n_places;
	auto const n_ships = parameters.n_ships;

	// create scip model
	auto model = scip::Model::prob_basic();
	model.set_name(fmt::format("SAP-{}-{}-{}", parameters.n_months, parameters.n_places, parameters.n_ships));
	auto* const scip = model.get_scip_ptr();
	scip::call(SCIPsetObjsense, scip, SCIP_OBJSENSE_MAXIMIZE);

	// sample coefficients
	xt::xarray<size_t>::shape_type shape = {n_months, n_places, n_ships};

	// sample returns
	xt::xtensor<SCIP_Real, 3> availability = xt::random::binomial<size_t>(shape, 1, 0.8);
	xt::xtensor<SCIP_Real, 3> returns = xt::random::rand<size_t>(shape, 3, 9);

	// auto v = xt::filter(returns, availability >= 5);
	// v = 0;
	// filtration(returns, availability == 0) = 0;
	returns *= availability;

	// sample Y_ijk - contraints for each ship at every place
	xt::xtensor<SCIP_Real, 3> max_prod = xt::random::rand<size_t>(shape, 10, 60);
	xt::xtensor<SCIP_Real, 2> month_constr = xt::random::rand<size_t>(shape, 30, 60);
	auto m_c = xt::expand_dims(month_constr, 2);
	max_prod = xt::clip(max_prod, 0, m_c);

	// sample E_ij month constaints
	xt::xtensor<SCIP_Real, 2> place_avail = xt::random::binomial<size_t>({n_months, n_places}, 1, 0.75);
	month_constr *= place_avail;

	// annual constaints
	xt::xtensor<SCIP_Real, 1> annual_constr = xt::random::rand<size_t>({n_places}, 250, 350);

	// add variables and constraints
	auto const vars = add_vars(scip, returns);
	add_constaints(scip, vars, max_prod, month_constr, annual_constr);

	return model;

}  // generate_instance

}  // namespace ecole::instance
