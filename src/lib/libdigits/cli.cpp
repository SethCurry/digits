#include <string>
#include <unordered_map>
#include <vector>
#include "./include/libdigits.hpp"
#include <iostream>



using string_map = std::unordered_map<std::string, std::vector<std::string>>;
template <class Keyword>
string_map generic_parse(std::vector<std::string> as, Keyword keyword)
{
    string_map result;

    std::string flag;
    for(auto&& x : as)
    {
        auto f = keyword(x);
        if(f.empty())
        {
            result[flag].push_back(x);
        }
        else
        {
            flag = f.front();
            result[flag]; // Ensure the flag exists
            flag = f.back();
        }
    }
    return result;
}


void parse_cli_flags(int argc, const char* argv[], Flags *flags)
{
    std::vector<std::string> args(argv + 1, argv + argc);


    std::unordered_map<std::string, std::string> keys = {
        {"--interval", "--interval"},
        {"--format", "--format"}
    };

    auto result = generic_parse(args, [&](auto&& s) -> std::vector<std::string> {
        if(keys.count(s) > 0)
            return {keys[s]};
        else
            return {};
    });

    auto intervalString = result.find("--interval");
    if (intervalString._M_cur != NULL) {
        auto toParse = intervalString->second.front().data();

        auto interval = std::stol(toParse);
        flags->interval = interval;
    }

    auto formatString = result.find("--format");
    if (formatString._M_cur != NULL) {
      auto format = std::string(formatString->second.front().data());

      std::cerr << format << std::endl;

      if (format.compare("full") == 0) {
          std::cerr << "Using full format" << std::endl;
        flags->format = 0;
      }

      if (format.compare("model") == 0) {
        std::cerr << "Using model format" << std::endl;
          flags->format = 1;
      };
    }
}
