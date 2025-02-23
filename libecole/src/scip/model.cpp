#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>

#include <fmt/format.h>
#include <range/v3/view/move.hpp>
#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "ecole/scip/callback.hpp"
#include "ecole/scip/exception.hpp"
#include "ecole/scip/model.hpp"
#include "ecole/scip/scimpl.hpp"
#include "ecole/scip/utils.hpp"
#include "ecole/utility/unreachable.hpp"

namespace ecole::scip {

Model::Model() : Model{std::make_unique<Scimpl>()} {
	scip::call(SCIPincludeDefaultPlugins, get_scip_ptr());
}

Model::Model(Model&&) noexcept = default;

Model::Model(std::unique_ptr<Scimpl>&& other_scimpl) : scimpl(std::move(other_scimpl)) {
	set_messagehdlr_quiet(true);
}

Model::~Model() = default;

Model& Model::operator=(Model&&) noexcept = default;

SCIP* Model::get_scip_ptr() noexcept {
	return scimpl->get_scip_ptr();
}
SCIP const* Model::get_scip_ptr() const noexcept {
	return scimpl->get_scip_ptr();
}

Model Model::copy() const {
	return std::make_unique<Scimpl>(scimpl->copy());
}

Model Model::copy_orig() const {
	return std::make_unique<Scimpl>(scimpl->copy_orig());
}

bool Model::operator==(Model const& other) const noexcept {
	return scimpl == other.scimpl;
}

bool Model::operator!=(Model const& other) const noexcept {
	return !(*this == other);
}

Model Model::from_file(std::filesystem::path const& filename) {
	auto model = Model{};
	model.read_problem(filename.c_str());
	return model;
}

Model Model::prob_basic(std::string const& name) {
	auto model = Model{};
	scip::call(SCIPcreateProbBasic, model.get_scip_ptr(), name.c_str());
	return model;
}

void Model::write_problem(std::filesystem::path const& filename) const {
	scip::call(SCIPwriteOrigProblem, const_cast<SCIP*>(get_scip_ptr()), filename.c_str(), nullptr, true);
}

void Model::read_problem(std::string const& filename) {
	scip::call(SCIPreadProb, get_scip_ptr(), filename.c_str(), nullptr);
}

void Model::set_messagehdlr_quiet(bool quiet) noexcept {
	SCIPsetMessagehdlrQuiet(get_scip_ptr(), static_cast<SCIP_Bool>(quiet));
}

std::string Model::name() const noexcept {
	return SCIPgetProbName(const_cast<SCIP*>(get_scip_ptr()));
}

void Model::set_name(std::string const& name) {
	scip::call(SCIPsetProbName, get_scip_ptr(), name.c_str());
}

SCIP_STAGE Model::stage() const noexcept {
	return SCIPgetStage(const_cast<SCIP*>(get_scip_ptr()));
}

ParamType Model::get_param_type(std::string const& name) const {
	auto* scip_param = SCIPgetParam(const_cast<SCIP*>(get_scip_ptr()), name.c_str());
	if (scip_param == nullptr) {
		throw scip::ScipError::from_retcode(SCIP_PARAMETERUNKNOWN);
	}
	switch (SCIPparamGetType(scip_param)) {
	case SCIP_PARAMTYPE_BOOL:
		return ParamType::Bool;
	case SCIP_PARAMTYPE_INT:
		return ParamType::Int;
	case SCIP_PARAMTYPE_LONGINT:
		return ParamType::LongInt;
	case SCIP_PARAMTYPE_REAL:
		return ParamType::Real;
	case SCIP_PARAMTYPE_CHAR:
		return ParamType::Char;
	case SCIP_PARAMTYPE_STRING:
		return ParamType::String;
	default:
		utility::unreachable();
	}
}

template <> void Model::set_param<ParamType::Bool>(std::string const& name, bool value) {
	scip::call(SCIPsetBoolParam, get_scip_ptr(), name.c_str(), value);
}
template <> void Model::set_param<ParamType::Int>(std::string const& name, int value) {
	scip::call(SCIPsetIntParam, get_scip_ptr(), name.c_str(), value);
}
template <> void Model::set_param<ParamType::LongInt>(std::string const& name, SCIP_Longint value) {
	scip::call(SCIPsetLongintParam, get_scip_ptr(), name.c_str(), value);
}
template <> void Model::set_param<ParamType::Real>(std::string const& name, SCIP_Real value) {
	scip::call(SCIPsetRealParam, get_scip_ptr(), name.c_str(), value);
}
template <> void Model::set_param<ParamType::Char>(std::string const& name, char value) {
	scip::call(SCIPsetCharParam, get_scip_ptr(), name.c_str(), value);
}
template <> void Model::set_param<ParamType::String>(std::string const& name, std::string const& value) {
	scip::call(SCIPsetStringParam, get_scip_ptr(), name.c_str(), value.c_str());
}

template <> bool Model::get_param<ParamType::Bool>(std::string const& name) const {
	SCIP_Bool value{};
	scip::call(SCIPgetBoolParam, const_cast<SCIP*>(get_scip_ptr()), name.c_str(), &value);
	return static_cast<bool>(value);
}
template <> int Model::get_param<ParamType::Int>(std::string const& name) const {
	int value{};
	scip::call(SCIPgetIntParam, const_cast<SCIP*>(get_scip_ptr()), name.c_str(), &value);
	return value;
}
template <> SCIP_Longint Model::get_param<ParamType::LongInt>(std::string const& name) const {
	SCIP_Longint value{};
	scip::call(SCIPgetLongintParam, const_cast<SCIP*>(get_scip_ptr()), name.c_str(), &value);
	return value;
}
template <> SCIP_Real Model::get_param<ParamType::Real>(std::string const& name) const {
	SCIP_Real value{};
	scip::call(SCIPgetRealParam, const_cast<SCIP*>(get_scip_ptr()), name.c_str(), &value);
	return value;
}
template <> char Model::get_param<ParamType::Char>(std::string const& name) const {
	char value{};
	scip::call(SCIPgetCharParam, const_cast<SCIP*>(get_scip_ptr()), name.c_str(), &value);
	return value;
}
template <> std::string Model::get_param<ParamType::String>(std::string const& name) const {
	char* ptr{};
	scip::call(SCIPgetStringParam, const_cast<SCIP*>(get_scip_ptr()), name.c_str(), &ptr);
	return ptr;
}

void Model::set_params(std::map<std::string, Param> name_values) {
	for (auto&& [name, value] : ranges::views::move(name_values)) {
		set_param(name, std::move(value));
	}
}

namespace {

nonstd::span<SCIP_PARAM*> get_params_span(Model const& model) noexcept {
	auto* const scip = const_cast<SCIP*>(model.get_scip_ptr());
	return {SCIPgetParams(scip), static_cast<std::size_t>(SCIPgetNParams(scip))};
}

}  // namespace

std::map<std::string, Param> Model::get_params() const {
	std::map<std::string, Param> name_values{};
	for (auto* const param : get_params_span(*this)) {
		auto name = std::string{SCIPparamGetName(param)};
		auto value = get_param<Param>(name);
		name_values.insert({std::move(name), std::move(value)});
	}
	return name_values;
}

void Model::disable_presolve() {
	scip::call(SCIPsetPresolving, get_scip_ptr(), SCIP_PARAMSETTING_OFF, true);
}
void Model::disable_cuts() {
	scip::call(SCIPsetSeparating, get_scip_ptr(), SCIP_PARAMSETTING_OFF, true);
}

nonstd::span<SCIP_VAR*> Model::variables() const noexcept {
	auto* const scip_ptr = const_cast<SCIP*>(get_scip_ptr());
	return {SCIPgetVars(scip_ptr), static_cast<std::size_t>(SCIPgetNVars(scip_ptr))};
}


std::pair<std::map<std::string, SCIP_Real>, std::size_t> Model::get_variables() const {
	std::map<std::string, SCIP_Real> name_values{};
	for (auto* const var : variables()) {
		auto name = std::string{SCIPvarGetName(var)};
		auto value = var->obj;
		name_values.insert({std::move(name), std::move(value)});
	}

	auto* const scip_ptr = const_cast<SCIP*>(get_scip_ptr());

	return {name_values, static_cast<std::size_t>(SCIPgetNVars(scip_ptr))};
}

nonstd::span<SCIP_VAR*> Model::lp_branch_cands() const {
	int n_vars = 0;
	SCIP_VAR** vars = nullptr;
	scip::call(
		SCIPgetLPBranchCands, const_cast<SCIP*>(get_scip_ptr()), &vars, nullptr, nullptr, &n_vars, nullptr, nullptr);
	return {vars, static_cast<std::size_t>(n_vars)};
}

nonstd::span<SCIP_VAR*> Model::pseudo_branch_cands() const {
	int n_vars = 0;
	SCIP_VAR** vars = nullptr;
	scip::call(SCIPgetPseudoBranchCands, const_cast<SCIP*>(get_scip_ptr()), &vars, &n_vars, nullptr);
	return {vars, static_cast<std::size_t>(n_vars)};
}

nonstd::span<SCIP_COL*> Model::lp_columns() const {
	auto* const scip_ptr = const_cast<SCIP*>(get_scip_ptr());
	if (SCIPgetStage(scip_ptr) != SCIP_STAGE_SOLVING) {
		throw ScipError::from_retcode(SCIP_INVALIDCALL);
	}
	return {SCIPgetLPCols(scip_ptr), static_cast<std::size_t>(SCIPgetNLPCols(scip_ptr))};
}

nonstd::span<SCIP_CONS*> Model::constraints() const noexcept {
	auto* const scip_ptr = const_cast<SCIP*>(get_scip_ptr());
	return {SCIPgetConss(scip_ptr), static_cast<std::size_t>(SCIPgetNConss(scip_ptr))};
}

nonstd::span<SCIP_ROW*> Model::lp_rows() const {
	auto* const scip_ptr = const_cast<SCIP*>(get_scip_ptr());
	if (SCIPgetStage(scip_ptr) != SCIP_STAGE_SOLVING) {
		throw ScipError::from_retcode(SCIP_INVALIDCALL);
	}
	return {SCIPgetLPRows(scip_ptr), static_cast<std::size_t>(SCIPgetNLPRows(scip_ptr))};
}

std::size_t Model::nnz() const noexcept {
	return static_cast<std::size_t>(SCIPgetNNZs(const_cast<SCIP*>(get_scip_ptr())));
}

void Model::transform_prob() {
	scip::call(SCIPtransformProb, get_scip_ptr());
}

void Model::presolve() {
	scip::call(SCIPpresolve, get_scip_ptr());
}

void Model::solve() {
	scip::call(SCIPsolve, get_scip_ptr());
}

bool Model::is_solved() const noexcept {
	return SCIPgetStage(const_cast<SCIP*>(get_scip_ptr())) == SCIP_STAGE_SOLVED;
}

SCIP_Real Model::primal_bound() const noexcept {
	auto* const scip = const_cast<SCIP*>(get_scip_ptr());
	switch (SCIPgetStage(scip)) {
	case SCIP_STAGE_TRANSFORMED:
	case SCIP_STAGE_INITPRESOLVE:
	case SCIP_STAGE_PRESOLVING:
	case SCIP_STAGE_EXITPRESOLVE:
	case SCIP_STAGE_PRESOLVED:
	case SCIP_STAGE_INITSOLVE:
	case SCIP_STAGE_SOLVING:
	case SCIP_STAGE_SOLVED:
		return SCIPgetPrimalbound(scip);
	default:
		return SCIPinfinity(scip);
	}
}

SCIP_Real Model::dual_bound() const noexcept {
	auto* const scip = const_cast<SCIP*>(get_scip_ptr());
	switch (SCIPgetStage(scip)) {
	case SCIP_STAGE_TRANSFORMED:
	case SCIP_STAGE_INITPRESOLVE:
	case SCIP_STAGE_PRESOLVING:
	case SCIP_STAGE_EXITPRESOLVE:
	case SCIP_STAGE_PRESOLVED:
	case SCIP_STAGE_INITSOLVE:
	case SCIP_STAGE_SOLVING:
	case SCIP_STAGE_SOLVED:
		return SCIPgetDualbound(scip);
	default:
		return -SCIPinfinity(scip);
	}
}

auto Model::solve_iter(nonstd::span<callback::DynamicConstructor const> arg_packs)
	-> std::optional<callback::DynamicCall> {
	return scimpl->solve_iter(arg_packs);
}

auto Model::solve_iter(callback::DynamicConstructor arg_pack) -> std::optional<callback::DynamicCall> {
	return solve_iter({&arg_pack, 1});
}

auto Model::solve_iter_continue(SCIP_RESULT result) -> std::optional<callback::DynamicCall> {
	return scimpl->solve_iter_continue(result);
}

}  // namespace ecole::scip
