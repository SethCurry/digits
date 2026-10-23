/******************************************************************************\
* Copyright (C) 2012-2014 Leap Motion, Inc. All rights reserved.               *
* Leap Motion proprietary and confidential. Not for distribution.              *
* Use subject to the terms of the Leap Motion SDK Agreement available at       *
* https://developer.leapmotion.com/sdk_agreement, or another agreement         *
* between Leap Motion and you, your company or other organization.             *
\******************************************************************************/

#include <chrono>
#include <iostream>
#include "Leap.h"
#include "libdigits.hpp"

using namespace Leap;


int main(int argc, const char** argv) {
  Flags cliFlags = {50, 0};

  parse_cli_flags(argc, argv, &cliFlags);

  // Create a sample listener and controller
  LibdigitsListener listener;

  std::cerr << "Setting interval to " << cliFlags.interval << std::endl;
  listener.setInterval(cliFlags.interval);
  std::cerr << "Setting format to " << cliFlags.format << std::endl;
  listener.setFormat(cliFlags.format);

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
