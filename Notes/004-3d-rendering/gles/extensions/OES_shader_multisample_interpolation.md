# OES_shader_multisample_interpolation

Name

    OES_shader_multisample_interpolation

Name Strings

    GL_OES_shader_multisample_interpolation

Contact

    Daniel Koch, NVIDIA Corporation (dkoch 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA
    Eric Werness, NVIDIA
    Graeme Leese, Broadcom
    Contributors to ARB_gpu_shader5
    Members of the OpenGL ES working group

Notice

    Copyright (c) 2010-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Complete. 
    Ratified by the Khronos Board of Promoters on 2014/03/14.

Version

    Last Modified Date:         February 11, 2014
    Revision:                   10

Number

    OpenGL ES Extension #172

Dependencies

    OpenGL ES 3.0 and GLSL ES 3.00 required.

    This extension is written against the OpenGL ES 3.0.2 (April 8, 2013)
    and the OpenGL ES Shading Language Specification Revision 4
    (March 6, 2013) specifications.

    This extension requires OES_sample_variables.

    This extension interacts with OES_sample_shading.

Overview

    In standard multisample rendering, an implementation is allowed to
    assign the same sets of fragment shader input values to each sample.
    This can cause aliasing where the fragment shader input values are
    used to generate a result that doesn't antialias itself, for example
    with alpha-tested transparency.

    This extension adds the "sample" qualifier that can be used on vertex
    outputs and fragment inputs. When the "sample" qualifier is used, the
    fragment shader is invoked separately for each covered sample and
    all such qualified interpolants must be evaluated at the corresponding
    sample point.

    This extension provides built-in fragment shader functions to provide
    fine-grained control over interpolation, including interpolating a
    fragment shader input at a programmable offset relative to the pixel
    center, a specific sample number, or at the centroid.

IP Status

    No known IP claims.

New Procedures and Functions

    None

New Tokens

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, GetFloatv,
    and GetInteger64v:

        MIN_FRAGMENT_INTERPOLATION_OFFSET_OES           0x8E5B
        MAX_FRAGMENT_INTERPOLATION_OFFSET_OES           0x8E5C
        FRAGMENT_INTERPOLATION_OFFSET_BITS_OES          0x8E5D

Additions to Chapter 2 of the OpenGL ES 3.0 Specification
(OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL ES 3.0 Specification
(Rasterization)

    Modify Section 3.3, Multisampling, p. 95

    (add new paragraph as the second-last paragraph, at the end of the
    section, p. 96)

    If the value of SAMPLE_BUFFERS is one and the current program object
    includes a fragment shader with one or more input variables qualified with
    "sample in", the data associated with those variables will be assigned
    independently.  The values for each sample must be evaluated at the
    location of the sample.  The data associated with any other variables not
    qualified with "sample in" need not be evaluated independently for each
    sample.


    Modify Section 3.9.1, Shader Variables, p. 161

    (modify the second last paragraph of the section to be as follows, p. 161)

    When interpolating input variables, the default
    screen-space location at which these variables are sampled is defined in
    previous rasterization sections.  The default location may be overriden by
    interpolation qualifiers.  When interpolating variables declared using
    "centroid in", the variable is sampled at a location within the pixel
    covered by the primitive generating the fragment.  When interpolating
    variables declared using "sample in" when the value of SAMPLE_BUFFERS is
    one, the fragment shader will be invoked separately for each covered sample
    and the variable will be sampled at the corresponding sample point.

    (add the following paragraph immediately after the above, p. 161)

    Additionally, built-in fragment shader functions provide further
    fine-grained control over interpolation.  The built-in functions
    interpolateAtCentroid() and interpolateAtSample() will sample variables as
    though they were declared with the "centroid" or "sample" qualifiers,
    respectively.  The built-in function interpolateAtOffset() will sample
    variables at a specified (x,y) offset relative to the center of the pixel.
    The range and granularity of offsets supported by this function is
    implementation-dependent.  If either component of the specified offset is
    less than MIN_FRAGMENT_INTERPOLATION_OFFSET_OES or greater than
    MAX_FRAGMENT_INTERPOLATION_OFFSET_OES, the position used to interpolate the
    variable is undefined.  Not all values of <offset> may be supported; x and
    y offsets may be rounded to fixed-point values with the number of fraction
    bits given by the implementation-dependent constant
    FRAGMENT_INTERPOLATION_OFFSET_BITS_OES.

Additions to Chapter 4 of the OpenGL ES 3.0 Specification
(Per-Fragment Operations and the Framebuffer)

    None.

Additions to Chapter 5 of the OpenGL ES 3.0 Specification
(Special Functions)

    None.

Additions to Chapter 6 of the OpenGL ES 3.0 Specification
(State and State Requests)

    None.

Additions to Appendix A of the OpenGL ES 3.0 (Compatibility Profile)
Specification (Invariance)

    None.

Additions to the EGL Specifications

    None.

Modifications to The OpenGL ES Shading Language Specification, Version 3.00.04

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_OES_shader_multisample_interpolation : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL ES Shading Language:

      #define GL_OES_shader_multisample_interpolation  1


    Modify Section 3.7, Keywords, p. 15

    (add to the keyword list)

      sample

    (remove from the reserved keywords list)

      sample

    Modify Section 4.3, Storage Qualifiers, p. 33

    (add to first table on the page)

      Qualifier         Meaning
      --------------    ----------------------------------------
      sample in         linkage with per-sample interpolation
      sample out        linkage with per-sample interpolation

    (modify third paragraph, p. 33) These interpolation qualifiers may only
    precede the qualifiers in, centroid in, sample in, out, centroid out, or
    sample out in a declaration.  ...


    Modify Section 4.3.4, Input Variables, p. 35

    (modify first paragraph of section) Shader input variables are declared
    with the in, centroid in, or sample in storage qualifiers. ... Variables
    declared as in, centroid in, or sample in may not be written to during
    shader execution. ...

    (modify the last paragraph, p. 35) ... It is an error to use "centroid in",
    "sample in" or the interpolation qualifiers in a vertex shader input. ...

    (modify third paragraph, p. 36) ...  Fragment shader inputs get
    per-fragment values, typically interpolated from a previous stage's
    outputs.  They are declared in fragment shaders with the in, centroid in,
    or sample in storage qualifiers. ...

    (add to examples immediately below)

      sample in vec4 perSampleColor;

    (replace the last paragraph of section, specifying that "sample" only need
     be used in the fragment shader) The output of the vertex shader and the
     input of the fragment shader form an interface.  For this interface,
     vertex shader output variables and fragment shader input variables of the
     same name must match in type and qualification, with a few exceptions:
     The storage qualifiers must, of course, differ (one is in and one is
     out). Also, the "sample" qualifier may differ.  When the "sample"
     qualifier does not match, the qualifier declared (if any) in the vertex
     shader is ignored.

    Modify Section 4.3.6, Output Variables, p. 37

    (modify first paragraph of section) Shader output variables are declared
    with the out, centroid out, or sample out storage qualifiers. ...

    (modify third paragraph of section) Vertex output variables
    output per-vertex data and are declared using the out, centroid out, or
    sample out storage qualifiers. ...

    (add to examples immediately below)

      sample out vec4 perSampleColor;

    (modify last paragraph, p. 37) Fragment outputs output per-fragment data
    and are declared using the out storage qualifier. It is an error to use
    "centroid out" or "sample out" in a fragment shader. ...


    Modify Section 4.3.9, Interpolation, p. 44

    (modify first paragraph of section, add reference to sample in/out) The
    presence of and type of interpolation is controlled by the storage
    qualifiers centroid in, sample in, centroid out, and sample out, by the
    optional interpolation qualifiers smooth, and flat. When no interpolation
    qualifier is present, smooth interpolation is used. ...

    (modify second paragraph) ... A variable may be qualified as flat centroid
    or flat sample, which will mean the same thing as qualifying it only as
    flat.

    (replace last paragraph, p. 44)

    When single-sampling, or for fragment shader input
    variables qualified with neither "centroid in" nor "sample in", the value
    is interpolated to the pixel's center and
    a single value may be assigned to each sample within the pixel, to the
    extent permitted by the OpenGL ES Specification.

    When multisampling, "centroid" and "sample" may be
    used to control the location and frequency of the sampling of the
    qualified fragment shader input.  If a fragment shader input is qualified
    with "centroid", a single value may be assigned to that variable for all
    samples in the pixel, but that value must be interpolated at a location
    that lies in both the pixel and in the primitive being rendered, including
    any of the pixel's samples covered by the primitive.  Because the location
    at which the variable is interpolated may be different in neighboring pixels,
    and derivatives may be computed by computing differences in neighboring pixels,
    derivatives of centroid-sampled inputs may be less accurate than those for
    non-centroid interpolated variables.  If a fragment shader input is
    qualified with "sample", a separate value must be assigned to that
    variable for each covered sample in the pixel, and that value must be
    sampled at the location of the individual sample.


    Modify Section 8.9, Fragment Processing Functions, p. 99

    (add new functions to the end of section, p. 101)

    Built-in interpolation functions are available to compute an interpolated
    value of a fragment shader input variable at a shader-specified (x,y)
    location.  A separate (x,y) location may be used for each invocation of
    the built-in function, and those locations may differ from the default
    (x,y) location used to produce the default value of the input.
    For the interpolateAt* functions, the call will return a precision
    qualification matching the precision of the "interpolant" argument to
    the function call.

      float interpolateAtCentroid(float interpolant);
      vec2 interpolateAtCentroid(vec2 interpolant);
      vec3 interpolateAtCentroid(vec3 interpolant);
      vec4 interpolateAtCentroid(vec4 interpolant);

      float interpolateAtSample(float interpolant, int sample);
      vec2 interpolateAtSample(vec2 interpolant, int sample);
      vec3 interpolateAtSample(vec3 interpolant, int sample);
      vec4 interpolateAtSample(vec4 interpolant, int sample);

      float interpolateAtOffset(float interpolant, vec2 offset);
      vec2 interpolateAtOffset(vec2 interpolant, vec2 offset);
      vec3 interpolateAtOffset(vec3 interpolant, vec2 offset);
      vec4 interpolateAtOffset(vec4 interpolant, vec2 offset);

    The function interpolateAtCentroid() will return the value of the input
    <interpolant> sampled at a location inside the both the pixel and
    the primitive being processed.  The value obtained would be the same value
    assigned to the input variable if declared with the "centroid" qualifier.

    The function interpolateAtSample() will return the value of the input
    <interpolant> variable at the location of the sample numbered <sample>.  If
    multisample buffers are not available, the input varying will be evaluated
    at the center of the pixel.  If sample <sample> does
    not exist, the position used to interpolate the input varying is
    undefined.

    The function interpolateAtOffset() will return the value of the input
    <interpolant> variable sampled at an offset from the center of the pixel
    specified by <offset>.  The two floating-point components of <offset>
    give the offset in pixels in the x and y directions, respectively.
    An offset of (0,0) identifies the center of the pixel.  The range and
    granularity of offsets supported by this function is
    implementation-dependent.

    For all of the interpolation functions, <interpolant> must be an input
    variable or an element of an input variable declared as an array.
    Component selection operators (e.g., ".xy") may not be used when
    specifying <interpolant>.  If <interpolant> is declared with a "flat"
    qualifier, the interpolated value will have the same value everywhere for
    a single primitive, so the location used for the interpolation has no
    effect and the functions just return that same value.  If <interpolant>
    is declared with the "centroid" qualifier, the value returned by
    interpolateAtSample() and interpolateAtOffset() will be evaluated
    at the specified location, ignoring the location normally used with the
    "centroid" qualifier.


    Modify Section 9, Shading Language Grammar, p. 92

    !!! TBD !!!

Dependencies on OES_sample_shading

    This extension builds upon the per-sample shading support provided by
    OES_sample_shading to provide a new "sample" qualifier on a fragment
    shader input that forces per-sample shading, and specifies that the value
    of the input be evaluated per-sample.

    There is no interaction between the extensions, except that shaders using
    the features of this extension seem likely to use features from
    OES_sample_shading as well.

Errors

    None.

New State

    None.

New Implementation Dependent State

    Add to table 6.28 (Implementation Dependent Values)

                                               Min.
    Get Value               Type  Get Command  Value  Description                  Sec.
    ----------------------  ----  -----------  -----  --------------------------   -----
    MIN_FRAGMENT_INTERP-     R    GetFloatv    -0.5   furthest negative offset     3.9.1
      OLATION_OFFSET_OES                               for interpolateAtOffset()
    MAX_FRAGMENT_INTERP-     R    GetFloatv    +0.5   furthest positive offset     3.9.1
      OLATION_OFFSET_OES                               for interpolateAtOffset()
    FRAGMENT_INTERPOLATION_  Z+   GetIntegerv    4    subpixel bits for            3.9.1
      OFFSET_BITS_OES                                  interpolateAtOffset()

Issues

    (0) This extension is based on parts of ARB_gpu_shader5.  What are the
        major differences?

        1- rebased to ES 3.0
        2- edits for consistency with GL 4.4/GLSL 440 specs
        3- removed mention of 'noperspective' interpolation qualifier
        4- removed mention of multisample enable
        5- retained tighter interpolation requirements from ES3.0
        6- moved gl_SampleMaskIn to OES_sample_variables
        7- added precision statement for the interpolateAt* functions.

        For historical issues, please see Issues 1, 2, 9, 11, and 12 in
        ARB_gpu_shader5.

    (1) What should we call this extension?

        RESOLVED: It will be called OES_shader_multisample_interpolation
        since it allows the shader to explicitly control the interpolation
        used for multisample interpolation.

    (2) "sample" and "centroid" have been split out as "Auxiliary Storage
        Qualifiers" in GLSL 4.xx specs.  Should we follow the way the
        extension spec was written, or modify it to be more like GLSL 4.40?

        RESOLVED. Leave it as the original extension was written. The changes
        in the core spec are related to spec re-ordering which it makes no
        sense to try to replicate via an extension.

    (3) For Section 4.9.3 when multisample is disabled the ES 300 spec says
        "the value is interpolated at the pixel's center", however GLSL 440
        says "the value may be interpolated anywhere within the pixel.  Which
        behaviour do we want to incorporate?

        RESOLVED. We retain the tighter ES 300 requirement which specifies
        that the pixel's center must be used.

    (4) What to do with the MULTISAMPLE enable language?

        RESOLVED. OpenGL ES does not support MULTISAMPLE enable. This is
        determined by the buffer properties, aka the value of SAMPLE_BUFFERS.

    (5) Do we need precision qualifiers on the interpolatAt* functions?

        RESOLVED. Yes. A statement was added for the interpolateAt* functions
        to clarify that the the precision is taken to from the 'interpolant'
        argument to the functions.

    (6) Should the shading language built-ins have OES suffixes?

        RESOLVED: No. Per Bug 11637, the WG made a policy decision
        that GLSL ES identifiers imported without semantic change
        or subsetting as OES extensions from core GLSL do not carry
        suffixes. The #extension mechanism must still be used to
        enable the appropriate extension before the functionality can
        be used.

Revision History

    Rev.    Date      Author    Changes
    ----  ----------  --------  -----------------------------------------
    10    2014-02-12  dkoch     remove GLSL suffixes per Issue 6.
    9     2014-01-31  pbrown    Add language explicitly not requiring that
                                "sample" qualifier match between stages (bug
                                11189), where the fragment qualifier wins.
    8     2014-01-30  dkoch     rename to OES, clean editing notes
    7     2013-10-03  dkoch     rewrote overview and dependecies
    6     2013-09-16  dkoch     renamed to shader_multisample_interpolation
    5     2013-09-10  dkoch     Mark issue 5 as resolved.
    4     2013-09-09  dkoch     Require OES_sample_variables
                                Move gl_SampleMaskIn interaction base extension.
                                Add precision statement for interpolateAt fn's.
                                Add a couple new edits for ESSL spec.
    3     2013-09-03  gleese    Moved sample variables to OES_sample_variables
    2     2013-08-26  dkoch     resolved issues 2-4 & supporting edits.
    1     2013-08-13  dkoch     port to ES 3.0
    0     2013-08-07  dkoch     reduced ARB_gpu_shader5 to sample-related features
