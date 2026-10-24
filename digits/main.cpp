/******************************************************************************\
* Copyright (C) 2012-2014 Leap Motion, Inc. All rights reserved.               *
* Leap Motion proprietary and confidential. Not for distribution.              *
* Use subject to the terms of the Leap Motion SDK Agreement available at       *
* https://developer.leapmotion.com/sdk_agreement, or another agreement         *
* between Leap Motion and you, your company or other organization.             *
\******************************************************************************/

#include <chrono>
#include <iostream>
#include <ostream>
#include "Leap.h"
#include "lib/digits/include/libdigits.hpp"
#include "libdigits.hpp"
#include "CLI11.hpp"
#include <fstream>

using namespace Leap;

typedef void(CleanupFunction)();


int main(int argc, char** argv) {
    CLI::App app{"A CLI utility for capturing streams of frames from a Leap Motion device."};
    argv = app.ensure_utf8(argv);

    std::int32_t interval = 50;
    app.add_option("-i,--interval", interval, "The interval between capturing frames, in milliseconds");

    std::string format = "model";
    app.add_option("-f,--format", format, "The format to output in; either \"model\" or \"full\"");

    std::string outputStr = "stdout";
    app.add_option("-o,--output", outputStr, "The location to output to.  stdout will output to the terminal.");

    CLI11_PARSE(app, argc, argv);

    std::int32_t parsedFormat = 0;


    if (format == "full") {
        std::cerr << "Using full format" << std::endl;
        parsedFormat = 0;
    } else if (format == "model") {
        std::cerr << "Using model format" << std::endl;
        parsedFormat = 1;
    }


  // Create a sample listener and controller
  LibdigitsListener listener = LibdigitsListener(outputStr, parsedFormat, interval);



  Controller controller;

  // Have the sample listener receive events from the controller
  controller.addListener(listener);

  // Keep this process running until Enter is pressed
  std::cerr << "Press Enter to quit..." << std::endl;
  std::cin.get();

  // Remove the sample listener when done
  controller.removeListener(listener);

  return 0;
}
