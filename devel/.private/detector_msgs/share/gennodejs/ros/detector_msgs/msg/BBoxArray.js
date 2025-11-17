// Auto-generated. Do not edit!

// (in-package detector_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let BBox = require('./BBox.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class BBoxArray {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.bboxes = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('bboxes')) {
        this.bboxes = initObj.bboxes
      }
      else {
        this.bboxes = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type BBoxArray
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [bboxes]
    // Serialize the length for message field [bboxes]
    bufferOffset = _serializer.uint32(obj.bboxes.length, buffer, bufferOffset);
    obj.bboxes.forEach((val) => {
      bufferOffset = BBox.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type BBoxArray
    let len;
    let data = new BBoxArray(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [bboxes]
    // Deserialize array length for message field [bboxes]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.bboxes = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.bboxes[i] = BBox.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    object.bboxes.forEach((val) => {
      length += BBox.getMessageSize(val);
    });
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'detector_msgs/BBoxArray';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '7ed748f248b539ee6ad03e80ae91120d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Header header
    detector_msgs/BBox[] bboxes
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: detector_msgs/BBox
    string label
    float64 score
    int64 x
    int64 y
    int64 w
    int64 h
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new BBoxArray(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.bboxes !== undefined) {
      resolved.bboxes = new Array(msg.bboxes.length);
      for (let i = 0; i < resolved.bboxes.length; ++i) {
        resolved.bboxes[i] = BBox.Resolve(msg.bboxes[i]);
      }
    }
    else {
      resolved.bboxes = []
    }

    return resolved;
    }
};

module.exports = BBoxArray;
