# NV_shader_noperspective_interpolation

Name

    NV_shader_noperspective_interpolation

Name Strings

    GL_NV_shader_noperspective_interpolation

Contact

    Daniel Koch, NVIDIA (dkoch 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA
    Michael Chock, NVIDIA

Status

    Complete

Version

    Last Modified Date: October 24, 2014
    Revision: 2

Number

    OpenGL ES Extension #201

Dependencies

    OpenGL ES 3.0 and GLSL ES 3.00 are required.

    This specification is written against the OpenGL ES 3.1 (March 17,
    2014) and OpenGL ES 3.10 Shading Language (May 14, 2014) specifications.

    This extension interacts with OES_shader_multisample_interpolation.

    This extension trivially interacts with EXT_geometry_shader.

    This extension trivially interacts with EXT_tessellation_shader.

Overview

    In OpenGL 3.0 and later, and in other APIs, there are three types of
    interpolation qualifiers that are available for fragment shader inputs:
    flat, smooth, and noperspective.  The 'flat' qualifier indicates that no
    interpolation should be used. This is mandatory for integer-type
    variables. The 'smooth' qualifier indicates that interpolation should be
    performed in a perspective0correct manner. This is the default for
    floating-point type variables.  The 'noperspective' qualifier indicates
    that interpolation should be performed linearly in screen space.

    While perspective-correct (smooth) and non-interpolated (flat) are the
    two types of interpolation that most commonly used, there are important
    use cases for linear (noperspective) interpolation.  In particular, in
    some work loads where screen-space aligned geometry is common, the use of
    linear interpolation can result in performance and/or power improvements.

    The smooth and flat interpolation qualifiers are already supported in
    OpenGL ES 3.0 and later. This extension adds support for noperspective
    interpolation to OpenGL ES.

New Procedures and Functions

    None.

New Tokens

    None.

Additions to the OpenGL ES 3.1 Specification

    Modifications to Section 12.4.1 (Clipping Shader Outputs)

    (Insert a new paragraph as the second-to-last paragraph of the section)

    For vertex shader outputs specified to be interpolated without
    perspective correction (using the "noperspective" qualifier), the value
    of <t> used to obtain the output value associated with P will be adjusted
    to produce results that vary linearly in screen space.


    Modifications to Section 13.4.1 (Basic Line Segment Rasterization)

    (Replace the last paragraph of the section with the following language
    which adds in the description of noperspective interpolation)

    The "noperspective" and "flat" keywords used to declare shader outputs
    affect how they are interpolated.  When neither keyword is specified,
    interpolation is performed as described in equation 13.4.  When the
    "noperspective" keyword is specified, interpolation is performed in the
    same fashion as for depth values, as described in equation 13.5.  When
    the "flat" keyword is specified, no interpolation is performed, and
    outputs are taken from the corresponding input value of the provoking
    vertex corresponding to that primitive (see section 12.3).


    Modifications to Section 13.5.1 (Basic Polygon Rasterization)

    (Replace the paragraph which describes the interpolation keywords in
    the middle of p. 298)

    The "noperspective" and "flat" keywords used to declare shader outputs
    affect how they are interpolated.  When neither keyword is specified,
    interpolation is performed as described in equation 13.7.  When the
    "noperspective" keyword is specified, interpolation is performed in the
    same fashion as for depth values, as described in equation 13.8. When
    the "flat" keyword is specified, no interpolation is performed, and
    outputs are taken from the corresponding input value of the provoking
    vertex corresponding to that primitive (see section 12.3).


    Modifications to Section 13.5.3 (Polygon Multisample Rasterization)

    (replace the last paragraph of the section)

    The "noperspective" and "flat" qualifiers affect how shader outputs are
    interpolated in the same fashion as described for basic polygon
    rasterization in section 13.5.1.

Dependencies on OES_shader_multisample_interpolation

    If OES_shader_multisample_interpolation is not supported, ignore
    references to the interpolation functions in section 8.13.

Dependencies on EXT_geometry_shader

    If EXT_geometry_shader is supported the, noperspective keyword
    can be used on the outputs from geometry shaders.

    If EXT_geometry_shader is not support, ignore references to
    geometry shaders.

Dependencies on EXT_tessellation_shader

    If EXT_tessellation_shader is supported, the noperspective keyword
    can be used on the outputs from tessellation shaders.

    If EXT_tessellation_shader is not support, ignore references to
    tessellation shaders.

New State

    None.

Additions to the OpenGL ES Shading Language 3.10 Specification

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_NV_shader_noperspective_interpolation : <behavior>

    where <behavior> is as specified in section 3.4.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_NV_shader_noperspective_interpolation 1


    Modifications to Section 3.6 (Keywords):

    Remove "noperspective" from the list of reserved keywords and add it to
    the list of keywords.


    Modifications to Section 4.3 (Storage Qualifiers):

    (Add to the table of interpolation qualifiers)

    Qualifier          Meaning
    ---------          -------
    noperspective      linear interpolation


    Modifications to Section 4.3.4 (Input Variables):

    (Add to the list of fragment input examples:)

        noperspective in float temperature;
        noperspective centroid in vec2 myTexCoord;


    Modifications to Section 4.3.6 (Output Variables):

    (Add to the list of vertex (or tessellation or geometry) output examples:)

        noperspective out float temperature;
        noperspective centroid out vec2 myTexCoord;


    Modifications to Section 4.5 (Interpolation Qualifiers):

    (Add to the table of interpolation qualifiers:)

    Qualifier          Meaning
    ---------          -------
    noperspective      linear interpolation

    (Following the description of "smooth" add the following description:)

    "A variable qualified as "noperspective" must be interpolated linearly
    in screen space as described in equation 13.5 of the OpenGL ES Graphics
    System Specification, section 13.4 "Line Segments".


    Modifications to Section 8.13 (Fragment Processing Functions), as modified
    by OES_shader_multisample_interpolation:

    (Add to the end of the paragraph describing the interpolation functions)

    "For all the interpolation functions ...
    If <interpolant> is declared with the "noperspective" qualifier, the
    interpolated value will be computed without perspective correction."


    Modifications to Section 9 (Shading Language Grammar)

    (Add to the list of tokens returned from lexical analysis)

    NOPERSPECTIVE

    (update the interpolation_qualifier rule to be)

    interpolation_qualifier:
        SMOOTH
        FLAT
        NOPERSPECTIVE


Issues

    (1) Is this any different from the 'noperspective' functionality
    that was added in OpenGL 3.0?

    RESOLVED. No. This is intended to be identical and the language used
    for this specification was based on the deltas between GL 4.4 and
    ES 3.1.

    (2) What should we call this extension?

    RESOLVED: Options considered included:
     - shader_interpolation_noperspective
     - shader_noperspective_interpolation
    Using the second option as this is consistent with the naming of
    OES_shader_multisample_interpolation which added support for
    per-sample interpolation.

    (3) This is a small extension. Is there anything else we should add at
    the same time?

    RESOLVED. No. All the other related functionality is supported in ES
    or already has an extension.

Revision History

    Rev.    Date    Author    Changes
    ----  --------  --------- -------------------------------------------------
      1   06/05/14  dkoch     Initial draft based on GL 4.4 and GLSL 4.40
      2   10/24/14  dkoch     Mark complete, resolve issue.
