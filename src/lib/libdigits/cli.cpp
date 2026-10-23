#include "libdigits.hpp"

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using string_map = std::unordered_map<std::string, std::vector<std::string>>;

// Splits a list of argv-style tokens into a map keyed by recognized flags,
// where each value is the list of tokens that followed that flag. Unknown
// tokens are attached to the most recently seen flag (or "" before any flag).
template <class Keyword>
static string_map generic_parse(std::vector<std::string> args, Keyword keyword) {
    string_map result;

    std::string flag;
    for (auto&& token : args) {
        auto matched = keyword(token);
        if (matched.empty()) {
            result[flag].push_back(token);
        } else {
            flag = matched.front();
            result[flag]; // Ensure the flag key exists even with no values.
            flag = matched.back();
        }
    }
    return result;
}

void parse_cli_flags(int argc, const char* argv[], Flags& flags) {
    std::vector<std::string> args(argv + 1, argv + argc);

    const std::unordered_map<std::string, std::string> keys = {
        {"--interval", "--interval"},
        {"--format", "--format"},
    };

    const auto result = generic_parse(
        args, [&](const std::string& s) -> std::vector<std::string> {
            auto it = keys.find(s);
            if (it != keys.end()) {
                return {it->second};
            }
            return {};
        });

    if (auto it = result.find("--interval"); it != result.end()) {
        flags.interval = std::stol(it->second.front());
    }

    if (auto it = result.find("--format"); it != result.end()) {
        const std::string format = it->second.front();
        std::cerr << format << std::endl;

        if (format == "full") {
            std::cerr << "Using full format" << std::endl;
            flags.format = FORMAT_FULL;
        } else if (format == "model") {
            std::cerr << "Using model format" << std::endl;
            flags.format = FORMAT_MODEL;
        }
    }
}