# EXT_shader_integer_mix

Name

    EXT_shader_integer_mix

Name Strings

    GL_EXT_shader_integer_mix

Contact

    Matt Turner (matt.turner 'at' intel.com)

Contributors

    Matt Turner, Intel
    Ian Romanick, Intel

Status

    Shipping

Version

    Last Modified Date:         09/12/2013
    Author Revision:            6

Number

    OpenGL Extension #437
    OpenGL ES Extension #161 

Dependencies

    OpenGL 3.0 or OpenGL ES 3.0 is required. This extension interacts with
    GL_ARB_ES3_compatibility.

    This extension is written against the OpenGL 4.4 (core) specification
    and the GLSL 4.40 specification.

Overview

    GLSL 1.30 (and GLSL ES 3.00) expanded the mix() built-in function to
    operate on a boolean third argument that does not interpolate but
    selects. This extension extends mix() to select between int, uint,
    and bool components.

New Procedures and Functions

    None.

New Tokens

    None.

Additions to Chapter 8 of the GLSL 4.40 Specification (Built-in Functions)

    Modify Section 8.3, Common Functions

    Additions to the table listing common built-in functions:

      Syntax                       Description
      ---------------------------  --------------------------------------------------
      genIType mix(genIType x,     Selects which vector each returned component comes
                   genIType y,     from. For a component of a that is false, the
                   genBType a)     corresponding component of x is returned. For a
      genUType mix(genUType x,     component of a that is true, the corresponding
                   genUType y,     component of y is returned.
                   genBType a)
      genBType mix(genBType x,
                   genBType y,
                   genBType a)

Additions to the AGL/GLX/WGL Specifications

    None.

Modifications to The OpenGL Shading Language Specification, Version 4.40

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_EXT_shader_integer_mix : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_EXT_shader_integer_mix        1

Interactions with ARB_ES3_compatibility

    On desktop implementations that support ARB_ES3_compatibility,
    GL_EXT_shader_integer_mix can be enabled (and the new functions
    used) in shaders declared with '#version 300 es'.

GLX Protocol

    None.

Errors

    None.

New State

    None.

New Implementation Dependent State

    None.

Issues

    1) Should we allow linear interpolation of integers via a non-boolean
       third component?

    RESOLVED: No.

    2) Should we allow mix() to select between boolean components?

    RESOLVED: Yes. Implementing the same functionality using casts would be
    possible but ugly.

Revision History

    Rev.    Date      Author    Changes
    ----  --------    --------  ---------------------------------------------
      6   09/12/2013  idr       After discussions in Khronos, change vendor
                                prefix to EXT.

      5   09/09/2013  idr       Add ARB_ES3_compatibility interaction.

      4   09/06/2013  mattst88  Allow extension on OpenGL ES 3.0.

      3   08/28/2013  mattst88  Add #extension/#define changes.

      2   08/26/2013  mattst88  Change vendor prefix to MESA. Add mix() that
                                selects between boolean components.
      1   08/26/2013  mattst88  Initial revision
