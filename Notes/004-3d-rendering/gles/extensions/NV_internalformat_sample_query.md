# NV_internalformat_sample_query

Name

    NV_internalformat_sample_query

Name Strings

    GL_NV_internalformat_sample_query

Contact

    Daniel Koch, NVIDIA (dkoch 'at' nvidia.com)

Contributors

    Piers Daniell, NVIDIA
    Weiwan Liu, NVIDIA

Status

    Complete

Version

    Last Modified Date: October 10, 2014
    Revision: 2

Number

    OpenGL Extension #475
    OpenGL ES Extension #196

Dependencies

    This specification is written against the OpenGL 4.5 (Core Profile)
    Specification (September 19, 2014).

    OpenGL 4.2 or ARB_internalformat_query are required for an OpenGL
    implementation.

    OpenGL ES 3.0 is required for an OpenGL ES implementation.

    This extension interacts with OpenGL ES 3.1.

    This extension interacts with KHR_debug and OpenGL 4.3.

    This extension interacts with OES_texture_storage_multisample_2d_array.

Overview

    Some OpenGL implementations support modes of multisampling which have
    properties which are non-obvious to applications and/or which may not be
    standards conformant. The idea of non-conformant AA modes is not new,
    and is exposed in both GLX and EGL with config caveats and the
    GLX_NON_CONFORMANT_CONFIG for GLX and EGL_NON_CONFORMANT_CONFIG for EGL,
    or by querying the EGL_CONFORMANT attribute in newer versions of EGL.

    Both of these mechanisms operate on a per-config basis, which works as
    intended for window-based configs. However, with the advent of
    application-created FBOs, it is now possible to do all the multisample
    operations in an application-created FBO and never use a multisample
    window.

    This extension further extends the internalformat query mechanism
    (first introduced by ARB_internalformat_query and extended in
    ARB_internalformat_query2) and introduces a mechanism for a
    implementation to report properties of formats that may also be
    dependent on the number of samples.  This includes information
    such as whether the combination of format and samples should be
    considered conformant. This enables an implementation to report
    caveats which might apply to both window and FBO-based rendering
    configurations.

    Some NVIDIA drivers support multisample modes which are internally
    implemented as a combination of multisampling and automatic
    supersampling in order to obtain a higher level of anti-aliasing than
    can be directly supported by hardware. This extension allows those
    properties to be queried by an application with the MULTISAMPLES_NV,
    SUPERSAMPLE_SCALE_X_NV and SUPERSAMPLE_SCALE_Y_NV properties. For
    example, a 16xAA mode might be implemented by using 4 samples and
    up-scaling by a factor of 2 in each of the x- and y-dimensions.
    In this example, the driver might report MULTSAMPLES_NV of 4,
    SUPERSAMPLE_SCALE_X_NV of 2, SUPERSAMPLE_SCALE_Y_NV of 2 and
    CONFORMANT_NV of FALSE.


New Procedures and Functions

    void GetInternalformatSampleivNV(enum target, enum internalformat,
                                     sizei samples, enum pname,
                                     sizei bufSize, int *params);

New Types

    None.

New Tokens

    Accepted by the <target> parameter of GetInternalformatSampleivNV:

        RENDERBUFFER
        TEXTURE_2D_MULTISAMPLE
        TEXTURE_2D_MULTISAMPLE_ARRAY

    Accepted by the <pname> parameter of GetInternalformatSampleivNV:

        MULTISAMPLES_NV                         0x9371
        SUPERSAMPLE_SCALE_X_NV                  0x9372
        SUPERSAMPLE_SCALE_Y_NV                  0x9373
        CONFORMANT_NV                           0x9374


Additions to Chapter 22 of the OpenGL 4.5 (Core Profile) Specification
(Context State Queries)

    Add a new section 22.3.ifsq "Internal Format Sample Queries":

    Information about implementation-dependent support for sample related
    properties of internal formats can be queried with the command

        void GetInternalformatSampleivNV(enum target, enum internalformat,
                                         sizei samples, enum pname,
                                         sizei bufSize, int *params);

    <internalformat> must be color-renderable, depth-renderable, or
    stencil-renderable (as defined in section 9.4).

    <target> indicates the usage of the <internalformat>, and must be one of
    the targets that can be used for multisample resources, that is one of
    RENDERBUFFER, TEXTURE_2D_MULTISAMPLE, or TEXTURE_2D_MULTISAMPLE_ARRAY.

    <samples> indicates the number of samples of the <internalformat> for
    which properties are being queried. It is an error if the requested
    <samples> are not supported for requested <internalformat> and <target>.
    GetInternalformativ with the SAMPLES property can be used to determine
    if <samples> is supported.

    No more than <bufSize> integers will be written into <params>. If
    more data are available, they will be ignored and no error will be
    generated.

    <pname> indicates the information to query, and it is one of the
    following values. When a known property is queried, the associated
    value is written into <params>, otherwise <params> is unmodified.

    - MULTISAMPLES_NV: returns the number of multisamples used when a
      resource of the requested type and the specified <samples> is created.

    - SUPERSAMPLE_SCALE_X_NV: returns the super-sample scaling factor that
      is used in the X-dimension when a resource of the requested type and
      the specified <samples> is created.

    - SUPERSAMPLE_SCALE_Y_NV: returns the super-sample scaling factor that
      is used in the Y-dimension when a resource of the requested type and
      the specified <samples> is created.

    - CONFORMANT_NV: returns the conformance-compliance of a resource
      created with the requested type and the specified <samples>.
      TRUE is returned if the format/sample combination is supported in a
      compliant manner. FALSE is returned if the requested format/sample
      combination is not conformant. If this query reports
      non-conformant status and the debug output functionality is enabled,
      the GL will generate a debug output message describing the caveats.
      The message has the source DEBUG_SOURCE_API, the type
      DEBUG_TYPE_UNDEFINED_BEHAVIOR, and an implementation-dependent ID.

    Errors:
    The INVALID_ENUM error is generated if <target> is not one of
    RENDERBUFFER, TEXTURE_2D_MULTISAMPLE, or TEXTURE_2D_MULTISAMPLE_ARRAY.

    The INVALID_ENUM error is generated if <internalformat> is not
    color-, depth-, or stencil-renderable.

    The INVALID_ENUM error is generated if <pname> is not one of
    MULTISAMPLES_NV, SUPERSAMPLE_SCALE_X_NV, SUPERSAMPLE_SCALE_Y_NV, or
    CONFORMANT_NV.

    The INVALID_VALUE error is generated if <bufSize> is negative.

    The INVALID_OPERATION error is generated if a resource of the requested
    type and samples is not supported by the implementation.

Additions to the WGL/GLX/EGL Specification

    None.

Dependencies on OpenGL ES 3.1

    If OpenGL ES 3.1 is not supported in an OpenGL ES implementation,
    ignore references to TEXTURE_2D_MULTISAMPLE <target> and resources.

Dependencies on OES_texture_storage_multisample_2d_array.

    If OES_texture_storage_multisample_2d_array is not supported in an
    OpenGL ES implementation, ignore references to
    TEXTURE_2D_MULTISAMPLE_ARRAY. If the extension is supported, replace
    references to TEXTURE_2D_MULTISAMPLE_ARRAY with references to
    TEXTURE_2D_MULTISAMPLE_ARRAY_OES.

Dependencies on KHR_debug

    If KHR_debug or OpenGL 4.3 are not supported, ignore references to
    debug output functionality.  If KHR_debug is supported in an OpenGL ES
    context, append the _KHR suffix onto associated types.

New State

    None.

Sample Code

    // Obtain supported sample count for a format:
    GLint num_sample_counts = 0;
    GLenum ifmt = GL_RGBA8;
    GLenum target = GL_TEXTURE_2D_MULTISAMPLE;
    glGetInternalformativ(target, ifmt, NUM_SAMPLE_COUNTS, 1,
                          &num_sample_counts);

    // get the list of supported samples for this format
    GLint samples[num_sample_counts];
    glGetInternalformativ(target, ifmt, SAMPLES, num_sample_counts, samples);

    // loop over the supported formats and get per-sample properties
    for (int i=0; i<num_sample_counts; i++)
    {
        GLint multisample;
        GLint ss_scale_x, ss_scale_y;
        GLint conformant;
        glGetInternalformatSampleivNV(target, ifmt, samples[i],
                                      GL_MULTISAMPLES_NV,
                                      1, &multisample);
        glGetInternalformatSampleivNV(target, ifmt, samples[i],
                                      GL_SUPERSAMPLE_SCALE_X_NV,
                                      1, &ss_scale_x);
        glGetInternalformatSampleivNV(target, ifmt, samples[i],
                                      GL_SUPERSAMPLE_SCALE_Y_NV,
                                      1, &ss_scale_y);
        glGetInternalformatSampleivNV(target, ifmt, samples[i],
                                      GL_CONFORMANT_NV, 1, &conformant);
        // do something with this information :-)
    }


Conformance Tests

    TBD

Issues

    None yet!


Revision History

    Rev.    Date    Author    Changes
    ----  --------  --------  -----------------------------------------
     1    09/24/14  dkoch     Initial version
     2    10/10/14  weiwliu   Assign value to new tokens
