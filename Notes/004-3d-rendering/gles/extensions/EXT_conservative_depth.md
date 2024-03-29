# EXT_conservative_depth

Name

    EXT_conservative_depth

Name String

    GL_EXT_conservative_depth

Contact

    Tobias Hector, Imagination Technologies (tobias.hector 'at' imgtec.com)

Contributors

    Contributors to the original ARB_conservative_depth
    Ian Romanick
    Daniel Koch

Status

    Final

Version

    Last Modified Date:         14/09/2016
    Author Revision:            3

Number

    OpenGL ES Extension #268

Dependencies

    OpenGL ES 3.0 is required.

    This extension is written against the OpenGL ES Shading Language Version 3.00.6

Overview

    There is a common optimization for hardware accelerated implementation of
    OpenGL ES which relies on an early depth test to be run before the fragment
    shader so that the shader evaluation can be skipped if the fragment ends
    up being discarded because it is occluded.

    This optimization does not affect the final rendering, and is typically
    possible when the fragment does not change the depth programmatically.
    (i.e.: it does not write to the built-in gl_FragDepth output). There are,
    however a class of operations on the depth in the shader which could
    still be performed while allowing the early depth test to operate.

    This extension allows the application to pass enough information to the
    GL implementation to activate such optimizations safely.

New Procedures and Functions

    None.

New Tokens

    None.

Additions to Chapter 3 of the OpenGL ES Shading Language 3.00.06 Specification (Basics)

    Add a new Section 3.5.x, GL_EXT_conservative_depth Extension

    Including the following line in a shader can be used to control the language
    features described in this extension:

        #extension GL_EXT_conservative_depth: <behavior>

    where <behavior> is as described in section 3.5.

    A new preprocessor #define is added to the OpenGL Shading Language:

        #define GL_EXT_conservative_depth 1

Additions to Chapter 4 of the OpenGL Shading Language 3.00.06 Specification (Variables and Types)

    Modify Section 4.3.8.2 (Output Layout Qualifiers) page 47

    Modify the paragraph beginning: "Fragment shaders allow output layout
    qualifiers only..."

        Fragment shaders allow output layout qualifiers only on the interface
        out, or for the purposes of redeclaring the built-in variable
        gl_FragDepth (see Section 7.2, Fragment Shader Special Variables).

    Insert the following at the end of the section:

        The built-in fragment shader variable gl_FragDepth may be redeclared using
        one of the following layout qualifiers.

        layout-qualifier-id
            depth_any
            depth_greater
            depth_less
            depth_unchanged

        For example:

            layout (depth_greater) out float gl_FragDepth;

        The layout qualifier for gl_FragDepth specifies constraints on the final
        value of gl_FragDepth written by any shader invocation.  GL implementations
        may perform optimizations assuming that the depth test fails (or passes)
        for a given fragment if all values of gl_FragDepth consistent with the layout
        qualifier would fail (or pass).  If the final value of gl_FragDepth
        is inconsistent with its layout qualifier, the result of the depth test for
        the corresponding fragment is undefined.  However, no error will be
        generated in this case.  When the depth test passes and depth writes are
        enabled, the value written to the depth buffer is always the value of
        gl_FragDepth, whether or not it is consistent with the layout qualifier.

        By default, gl_FragDepth assumes the <depth_any> layout qualifier. When
        the layout qualifier for gl_FragDepth is <depth_any>, the shader compiler
        will note any assignment to gl_FragDepth modifying it in an unknown way,
        and depth testing will always be performed after the shader has executed.
        When the layout qualifier is "depth_greater", the GL will assume that the
        final value of gl_FragDepth is greater than or equal to the fragment's
        interpolated depth value, as given by the <z> component of gl_FragCoord.
        When the layout qualifier is <depth_less>, the GL will assume that any
        modification of gl_FragDepth will only decrease its value. When the
        layout qualifier is <depth_unchanged>, the shader compiler will honor
        any modification to gl_FragDepth, but the rest of the GL assume that
        gl_FragDepth is not assigned a new value.

        Redeclarations of gl_FragDepth are performed as follows:

            // redeclaration that changes nothing is allowed

            out float gl_FragDepth;

            // assume it may be modified in any way
            layout (depth_any) out float gl_FragDepth;

            // assume it may be modified such that its value will only increase
            layout (depth_greater) out float gl_FragDepth;

            // assume it may be modified such that its value will only decrease
            layout (depth_less) out float gl_FragDepth;

            // assume it will not be modified
            layout (depth_unchanged) out float gl_FragDepth;

        Within any shader, the first redeclarations of gl_FragDepth must appear
        before any use of gl_FragDepth. The built-in gl_FragDepth is only
        predeclared in fragment shaders, so redeclaring it in any other shader
        language will be illegal.

Revision History

    Rev.  Date        Author   Changes
    ----  ----------  -------  -------------------------------------------------
    1     08/09/2016  thector  Initial draft based on ARB_conservative_depth
    2     12/09/2016  thector  Converted to EXT as per feedback from Ian Romanick
    3     14/09/2016  thector  Removed reference to multiple fragment shaders as per Daniel Koch's feedback
