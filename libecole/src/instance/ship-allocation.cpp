#include <fmt/format.h>

#include <iostream>
#include <fstream>

#include <random>

#include <xtensor/xarray.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <xtensor/xnpy.hpp>
#include <xtensor/xio.hpp>

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

SCIP_VAR* add_var(SCIP * scip, std::string name, SCIP_Real cost, SCIP_Real lb, SCIP_Real ub, SCIP_Vartype vartype) {
	auto unique_var = scip::create_var_basic(scip, name.c_str(), lb, ub, cost, vartype);
	SCIP_VAR* var_ptr = unique_var.get();
	scip::call(SCIPaddVar, scip, var_ptr);
	return var_ptr;
}

xt::xtensor<SCIP_VAR*, 4> add_vars(SCIP* scip, xt::xtensor<SCIP_Real, 3> c) {
	xt::xtensor<SCIP_VAR*, 4>::shape_type shape = {static_cast<unsigned long>(3),
												   c.shape(0),
												   c.shape(1),
												   c.shape(2)};
	xt::xtensor<SCIP_VAR*, 4> vars(shape);

	for (size_t i = 0; i < shape[1]; ++i) {
		for (size_t j = 0; j < shape[2]; ++j) {
			for (size_t k = 0; k < shape[3]; ++k) {
				vars(0, i, j, k) = add_var(scip, fmt::format("x_{}_{}_{}", i, j, k), 0.0, 		 0, 1, SCIP_VARTYPE_BINARY);
				vars(1, i, j, k) = add_var(scip, fmt::format("q_{}_{}_{}", i, j, k), 0.0, 		 0, 1, SCIP_VARTYPE_CONTINUOUS);
				vars(2, i, j, k) = add_var(scip, fmt::format("y_{}_{}_{}", i, j, k), c(i, j, k), 0, 1, SCIP_VARTYPE_CONTINUOUS);
			}
		}
	}

	return vars;
}

auto add_constaints(
	SCIP * scip,
	xt::xtensor<SCIP_VAR*, 4> &vars,
	xt::xtensor<SCIP_Real, 3> &max_prod,
	xt::xtensor<SCIP_Real, 2> &month_constr,
	xt::xtensor<SCIP_Real, 1> &annual_constr,
	xt::xtensor<SCIP_Real, 2> &schedule) {

	// auto const pos_inf =  SCIPinfinity(scip);
	auto const neg_inf = -SCIPinfinity(scip);

	std::string name;

	auto quadcoefs = xt::xtensor<SCIP_Real, 1>({1}, 1.);
	auto lincoefs = xt::xtensor<SCIP_Real, 1>({1}, -1.);

	for (size_t j = 0; j < vars.shape(2); ++j) {
		for (size_t i = 0; i < vars.shape(1); ++i) {
			for (size_t k = 0; k < vars.shape(3); ++k) {
				// quadvars1 = {vars(0, i, j, k)};
				// quadvars2 = {vars(1, i, j, k)};
				// linvars   = {vars(2, i, j, k)};
				// size_t n_nonlinvars = 1;
				// size_t n_linvars    = 1;

				name = fmt::format("obj_to_cons_pos_{}_{}_{}_term", i, j, k);
				auto cons = scip::create_cons_non_linear(scip, name.c_str(),
													1, &vars(2, i, j, k), lincoefs.data(),
													1, &vars(0, i, j, k), &vars(1, i, j, k), quadcoefs.data(), 0, 0);
				scip::call(SCIPaddCons, scip, cons.get());

				// =======================================================================

				xt::xtensor<SCIP_VAR*, 1> x_ik = xt::view(vars, 0, i, xt::all(), k);
				name = fmt::format("S_{}_{}", i, k);
				auto coefs = xt::xtensor<SCIP_Real, 1>({vars.shape(2)}, 1.);
				cons = scip::create_cons_basic_linear(scip, name.c_str(),
													  x_ik.size(), &x_ik(0), coefs.data(), 0, schedule(i, k));
				scip::call(SCIPaddCons, scip, cons.get());
			}

			xt::xtensor<SCIP_VAR*, 1> x_ij = xt::view(vars, 2, i, j, xt::all());
			name = fmt::format("M_{}_{}", i, j);
			xt::xarray<SCIP_Real> coefs = xt::view(max_prod, i, j, xt::all());
			auto cons = scip::create_cons_basic_linear(
				scip, name.c_str(), x_ij.size(), &x_ij(0), &coefs(0), 0, month_constr(i, j));
			scip::call(SCIPaddCons, scip, cons.get());
		}

		xt::xtensor<SCIP_VAR*, 1> x_j = xt::flatten(xt::view(vars, 2, xt::all(), j, xt::all()));
		name = fmt::format("A_{}", j);
		xt::xarray<SCIP_Real> coefs = xt::flatten(xt::view(max_prod, xt::all(), j, xt::all()));
		auto cons = scip::create_cons_basic_linear(
			scip, name.c_str(), x_j.size(), &x_j(0), &coefs(0), 0, annual_constr(j));
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

	auto const r_path = parameters.r_path;
	auto const c_path = parameters.c_path;
	auto const m_path = parameters.m_path;
	auto const a_path = parameters.a_path;
	auto const s_path = parameters.s_path;

	// sample coefficients
	xt::xarray<size_t>::shape_type shape = {n_months, n_places, n_ships};

	// xt::xtensor<SCIP_Real, 3> returns = xt::load_npy<SCIP_Real>(r_path);
	// xt::xtensor<SCIP_Real, 3> max_prod = xt::load_npy<SCIP_Real>(c_path);
	// xt::xtensor<SCIP_Real, 2> month_constr = xt::load_npy<SCIP_Real>(m_path);
	// xt::xtensor<SCIP_Real, 1> annual_constr = xt::load_npy<SCIP_Real>(a_path);
	// xt::xtensor<SCIP_Real, 2> schedule = xt::load_npy<SCIP_Real>(s_path);

	xt::xarray<double> a = {1, 5, 9};
	auto b = xt::diag(a, 1);
	std::ofstream myfile;
	myfile.open("test.txt");
	myfile << b << std::endl;
	myfile.close();

	
	// sample returns
	xt::xtensor<SCIP_Real, 3> availability = xt::random::binomial<size_t>(shape, 1, 0.8, rng) + 0;
	xt::xtensor<SCIP_Real, 3> returns = xt::random::rand<SCIP_Real>(shape, 3, 9, rng) + 0;

	returns *= availability;

	// sample Y_ijk - contraints for each ship at every place
	xt::xtensor<SCIP_Real, 3> max_prod = xt::random::rand<SCIP_Real>(shape, 10, 60, rng) + 0;
	xt::xtensor<SCIP_Real, 2> month_constr = xt::random::rand<SCIP_Real>({n_months, n_places}, 30, 60, rng) + 0;
	auto m_c = xt::expand_dims(month_constr, 2);
	max_prod = xt::clip(max_prod, 0, m_c);

	// sample E_ij month constaints
	xt::xtensor<SCIP_Real, 2> place_avail = xt::random::binomial<size_t>({n_months, n_places}, 1, 0.75, rng) + 0;
	month_constr *= place_avail;

	// annual constaints
	std::uniform_real_distribution<double> unif(0.6, 0.9);
	std::default_random_engine re;
	double beta = unif(rng);
	xt::xtensor<SCIP_Real, 1> annual_constr = beta * xt::sum(month_constr, {0});

	xt::xtensor<SCIP_Real, 2> schedule = xt::random::binomial<size_t>({n_months, n_ships}, 1, 0.9, rng) + 0;

    xt::dump_npy("R.npy", returns);
    xt::dump_npy("C.npy", max_prod);
    xt::dump_npy("M.npy", month_constr);
    xt::dump_npy("A.npy", annual_constr);
    xt::dump_npy("S.npy", schedule);

    returns *= max_prod;

	// create scip model
	auto model = scip::Model::prob_basic();
	model.set_name(fmt::format("SAP-{}-{}-{}", parameters.n_months, parameters.n_places, parameters.n_ships));
	auto* const scip = model.get_scip_ptr();
	scip::call(SCIPsetObjsense, scip, SCIP_OBJSENSE_MAXIMIZE);

	// add variables and constraints
	xt::xtensor<SCIP_VAR*, 4> vars = add_vars(scip, returns);

	// // std::ofstream myfile;
	// // myfile.open("test.txt");
	// // auto shape_kek = vars.shape();
	// // for (auto& el : shape_kek) {myfile << el << ", "; }
	// // myfile.close();

	add_constaints(scip, vars, max_prod, month_constr, annual_constr, schedule);

	return model;

}  // generate_instance

}  // namespace ecole::instance
