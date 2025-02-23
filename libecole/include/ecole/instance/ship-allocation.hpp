#pragma once

#include <cstddef>

#include "ecole/export.hpp"
#include "ecole/instance/abstract.hpp"
#include "ecole/random.hpp"

namespace ecole::instance {

class ECOLE_EXPORT SAPGenerator : public InstanceGenerator {
public:
    struct ECOLE_EXPORT Parameters {
        std::string r_path = "";
        std::string c_path = "";
        std::string m_path = "";
        std::string a_path = "";
        std::string s_path = "";
        std::size_t n_months = 12;  // NOLINT(readability-magic-numbers)
        std::size_t n_places = 20;  // NOLINT(readability-magic-numbers)
        std::size_t n_ships = 35;   // NOLINT(readability-magic-numbers)
    };

    ECOLE_EXPORT static scip::Model generate_instance(Parameters parameters, RandomGenerator& rng);

    ECOLE_EXPORT SAPGenerator(Parameters parameters, RandomGenerator rng);
    ECOLE_EXPORT SAPGenerator(Parameters parameters);
    ECOLE_EXPORT SAPGenerator();

    ECOLE_EXPORT scip::Model next() override;
    ECOLE_EXPORT void seed(Seed seed) override;
    [[nodiscard]] ECOLE_EXPORT bool done() const override { return false; }

    [[nodiscard]] ECOLE_EXPORT Parameters const& get_parameters() const noexcept { return parameters; }

private:
    RandomGenerator rng;
    Parameters parameters;
};

}  // namespace ecole::instance
