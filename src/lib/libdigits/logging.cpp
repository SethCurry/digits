#include <iostream>
#include <cstring>
#include "Leap.h"
#include "../../include/json.hpp"
#include "./include/libdigits.hpp"

using namespace Leap;
using json = nlohmann::json;


std::vector<float> vecToVec(Vector vector) {
  std::vector<float> newVec = {vector.x, vector.y, vector.z};

  return newVec;
}

Vector avgVectors(const Vector &one, const Vector &two) {
  return Vector((one.x + two.x) / 2.0f,
                (one.y + two.y) / 2.0f,
                (one.z + two.z) / 2.0f);
}


// Builds the flat list of inputs used by the inference model, in the same
// order as SkeletalHGRFrame.to_frame() in digipy/training.py.
//
// For each finger in [thumb, index, middle, ring, pinky] we emit the
// first_knuckle, second_knuckle, third_knuckle and tip positions (each as
// x, y, z), followed by the palm and wrist positions. The knuckle positions
// are the midpoints between adjacent bone joints, matching the Python code.
std::vector<float> modelInputs(Hand hand) {
  std::vector<float> dim;
  dim.reserve(66);

  // Index fingers by type so we can emit them in a fixed order regardless
  // of the order Leap reports them in.
  Finger fingersByType[5];
  const FingerList fingers = hand.fingers();
  for (FingerList::const_iterator fl = fingers.begin(); fl != fingers.end(); ++fl) {
    const Finger finger = *fl;
    if (finger.type() >= 0 && finger.type() < 5) {
      fingersByType[finger.type()] = finger;
    }
  }

  for (int t = 0; t < 5; ++t) {
    const Finger &finger = fingersByType[t];

    Bone metacarpal = finger.bone(Bone::TYPE_METACARPAL);
    Bone proximal = finger.bone(Bone::TYPE_PROXIMAL);
    Bone middle = finger.bone(Bone::TYPE_INTERMEDIATE);
    Bone distal = finger.bone(Bone::TYPE_DISTAL);

    Vector firstKnuckle = avgVectors(proximal.prevJoint(), metacarpal.nextJoint());
    Vector secondKnuckle = avgVectors(middle.prevJoint(), proximal.nextJoint());
    Vector thirdKnuckle = avgVectors(distal.prevJoint(), middle.nextJoint());
    Vector tip = distal.nextJoint();

    for (const Vector &joint : {firstKnuckle, secondKnuckle, thirdKnuckle, tip}) {
      dim.push_back(joint.x);
      dim.push_back(joint.y);
      dim.push_back(joint.z);
    }
  }

  Vector palm = hand.palmPosition();
  dim.push_back(palm.x);
  dim.push_back(palm.y);
  dim.push_back(palm.z);

  Vector wrist = hand.arm().wristPosition();
  dim.push_back(wrist.x);
  dim.push_back(wrist.y);
  dim.push_back(wrist.z);

  return dim;
}

json boneToJson(Bone bone, Bone::Type boneType) {
    json jBone;


    jBone["name"] = boneNames[boneType];
    jBone["start"] = vecToVec(bone.prevJoint());
    jBone["end"] = vecToVec(bone.nextJoint());
    jBone["direction"] = vecToVec(bone.direction());

    return jBone;
}

json fingerToJson(Finger finger) {
    json jFinger;
    jFinger["name"] = fingerNames[finger.type()];
    jFinger["id"] = finger.id();
    jFinger["length"] = finger.length();
    jFinger["width"] = finger.width();

    std::vector<json> bonesVec = {};

    // Get finger bones
    for (int b = 0; b < 4; ++b) {
      Bone::Type boneType = static_cast<Bone::Type>(b);
      Bone bone = finger.bone(boneType);

      bonesVec.push_back(boneToJson(bone, boneType));
    }

    jFinger["bones"] = bonesVec;

    return jFinger;
}

json handToJson(Hand hand) {
  json j;

  j["is_left"] = hand.isLeft();
  j["id"] = hand.id();

  Vector palmPos = hand.palmPosition();
  std::vector<float> palmVec = {palmPos.x, palmPos.y, palmPos.z};
  j["palm_position"] = palmVec;

  const Vector normal = hand.palmNormal();
  const Vector direction = hand.direction();

  j["pitch"] = direction.pitch();
  j["roll"] = normal.roll();
  j["yaw"] = direction.yaw();

  // Get the Arm bone
  Arm arm = hand.arm();

  json jArm;

  Vector armDirection = arm.direction();
  std::vector<float> armVec = {armDirection.x, armDirection.y, armDirection.z};
  jArm["direction"] = armVec;
  jArm["wrist_position"] = vecToVec(arm.wristPosition());
  jArm["elbow_position"] = vecToVec(arm.elbowPosition());

  j["arm"] = jArm;

  std::vector<json> fingersVec = {};

  // Get fingers
  const FingerList fingers = hand.fingers();
  for (FingerList::const_iterator fl = fingers.begin(); fl != fingers.end();
       ++fl) {

    const Finger finger = *fl;

    fingersVec.push_back(fingerToJson(finger));
  }

  j["fingers"] = fingersVec;


  return j;
}

void logFrame(Frame frame) {
  if (frame.hands().count() < 1) {
    return;
  }
  json j;
  j["frame_id"] = frame.id();
  j["timestamp"] = frame.timestamp();
  j["num_hands"] = frame.hands().count();
  j["num_extended_fingers"] = frame.fingers().extended().count();
  j["num_gestures"] = frame.gestures().count();
  j["num_tools"] = frame.tools().count();

  std::vector<json> handsVec = {};

    HandList hands = frame.hands();     // Get the Arm bone
    for (HandList::const_iterator hl = hands.begin(); hl != hands.end(); ++hl) {
      // Get the first hand
      const Hand hand = *hl;
      const json newHand = handToJson(hand);
      handsVec.push_back(newHand);
    }

    j["hands"] = handsVec;

    std::cout << j.dump() << std::endl;
}

void logModelFrame(Frame frame) {
  if (frame.hands().count() < 1) {
    return;
  }

  // Emit one comma-separated line of model inputs per hand.
  HandList hands = frame.hands();
  for (HandList::const_iterator hl = hands.begin(); hl != hands.end(); ++hl) {
    const Hand hand = *hl;
    std::vector<float> inputs = modelInputs(hand);

    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << inputs[i];
    }
    std::cout << std::endl;
  }
}
