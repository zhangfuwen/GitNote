# EXT_draw_transform_feedback

Name

    EXT_draw_transform_feedback

Name Strings

    GL_EXT_draw_transform_feedback

Contact

    Mark Callow (khronos 'at' callow.im)

Notice

    Copyright (c) 2016 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete

IP Status

    No known IP issues

Version

    Date: Oct 14, 2016
    Revision: 2

Number

    OpenGL ES Extension #272

Dependencies

    Requires OpenGL ES 3.2.

    Written based on the wording of the OpenGL ES 3.2 specification
    (June 15th, 2016).

Overview

    This extension adds the DrawTransformFeedback commands. These
    were omitted from OpenGL ES 3.0 because the number of vertices
    captured by transform feedback could never be different than
    the number drawn during capture. The addition of geometry shaders
    in OpenGL ES 3.2 broke that assumption but, due to an oversight,
    DrawTransformFeedback et al were not reinstated. The
    DrawTransformFeedback commands unlock the full potential of
    geometry shaders.

Issues

    None

New Procedures and Functions

   void DrawTransformFeedbackEXT( enum mode, uint id );
   void DrawTransformFeedbackInstancedEXT( enum mode, uint id, sizei instancecount );

New Tokens

    None

Additions to Chapter 1-11 of the OpenGL ES 3.2 specification

    None

Additions to Chapter 12 of the OpenGL 3.2 Specification (Fixed-
Function Vertex Post-Processing)

    Add new section 12.1.3 Transform Feedback Draw Operations

    When transform feedback is active, the values of output variables
    or transformed vertex attributes are captured into the buffer
    objects attached to the current transform feedback object.
    After transform feedback is complete, subsequent rendering
    operations may use the contents of these buffer objects (see
    section 6). The number of vertices captured during transform
    feedback is stored in the transform feedback object and may be
    used in conjunction with the commands

        void DrawTransformFeedbackEXT( enum mode, uint id );
        void DrawTransformFeedbackInstancedEXT( enum mode, uint id,
           sizei instancecount );

    to replay the captured vertices.
        Calling DrawTransformFeedbackInstancedEXT is equivalent
    to calling DrawArraysInstanced with mode as specified, first
    set to zero, count set to the number of vertices captured the
    last time transform feedback was active on the transform feedback
    object named id, and instancecount as specified.
        Calling DrawTransformFeedbackEXT is equivalent to calling
    DrawTransformFeedbackInstancedEXT with instancecount set to
    one.
        Note that the vertex count is from the number of vertices
    recorded during the transform feedback operation. If no
    outputs are recorded, the corresponding vertex count will be
    zero.

    No error is generated if the transform feedback object named
    by id is active; the vertex count used for the rendering operation
    is set by the previous EndTransformFeedback command.

    Errors

    An INVALID_VALUE error is generated if id is not the name of a
    transform feedback object.

    An INVALID_VALUE error is generated if instancecount is negative.

    An INVALID_OPERATION error is generated if EndTransformFeedback
    has never been called while the object named by id was bound.

Additions to Chapters 13-21 of the OpenGL 3.2 Specification

    None

Additions to the EGL/AGL/GLX/WGL Specifications

    None

GLX Protocol

    n/a

New State

    None

Revision History

    Rev.  Date     Author     Changes
    ----  -------- ---------  -----------------------------------------
      0   09/14/16 markc      Initial draft.
      1   09/27/16 markc      Add suffices. Remove vestiges of
                              "stream". Change overview to plural.
      2   10/14/16 markc      Change to EXT.

# vim:ai:ts=4:sts=4:sw=4:expandtab:textwidth=70
