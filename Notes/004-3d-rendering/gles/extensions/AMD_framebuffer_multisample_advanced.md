# AMD_framebuffer_multisample_advanced

Name

    AMD_framebuffer_multisample_advanced

Name Strings

    GL_AMD_framebuffer_multisample_advanced

Contact

    Marek Olsak, AMD (marek.olsak 'at' amd.com)

Status

    Complete.

Version

    Last Modified Date:  June 28, 2018
    Revision #1

Number

    OpenGL Extension #523
    OpenGL ES Extension #303

Dependencies

    OpenGL dependencies:

        Requires GL_ARB_framebuffer_object.

    OpenGL ES dependencies:

        Requires OpenGL ES 3.0.

    This extension is written against the OpenGL 4.5 (Core Profile)
    specification.

Overview

    This extension extends ARB_framebuffer_object by allowing compromises
    between image quality and memory footprint of multisample
    antialiasing.

    ARB_framebuffer_object introduced RenderbufferStorageMultisample
    as a method of defining the parameters for a multisample render
    buffer. This function takes a <samples> parameter that has strict
    requirements on behavior such that no compromises in the final image
    quality are allowed. Additionally, ARB_framebuffer_object requires
    that all framebuffer attachments have the same number of samples.

    This extension extends ARB_framebuffer_object by providing a new
    function, RenderbufferStorageMultisampleAdvancedAMD, that
    distinguishes between samples and storage samples for color
    renderbuffers where the number of storage samples can be less than
    the number of samples. This extension also allows non-matching sample
    counts between color and depth/stencil renderbuffers.

    This extension does not require any specific combination of sample
    counts to be supported.

IP Status

    No known IP issues.

New Procedures and Functions

    void RenderbufferStorageMultisampleAdvancedAMD(
             enum target, sizei samples, sizei storageSamples,
             enum internalformat, sizei width, sizei height );

    void NamedRenderbufferStorageMultisampleAdvancedAMD(
             uint renderbuffer, sizei samples, sizei storageSamples,
             enum internalformat, sizei width, sizei height );

New Tokens

    Accepted by the <pname> parameter of GetRenderbufferParameteriv:

        RENDERBUFFER_STORAGE_SAMPLES_AMD            0x91B2

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    GetInteger64v, GetFloatv, GetDoublev:

        MAX_COLOR_FRAMEBUFFER_SAMPLES_AMD           0x91B3
        MAX_COLOR_FRAMEBUFFER_STORAGE_SAMPLES_AMD   0x91B4
        MAX_DEPTH_STENCIL_FRAMEBUFFER_SAMPLES_AMD   0x91B5
        NUM_SUPPORTED_MULTISAMPLE_MODES_AMD         0x91B6
        SUPPORTED_MULTISAMPLE_MODES_AMD             0x91B7

Additions to Chapter 9 of the OpenGL 4.5 (Core Profile) Specification
(Framebuffers and Framebuffer Objects)

    In section 9.2.3.1, "Multisample Queries", remove the last paragraph
    beginning with "Otherwise" and add:

    Otherwise, the value of SAMPLES is equal to the value of
    RENDERBUFFER_SAMPLES or TEXTURE_SAMPLES (depending on the type of
    attachments) of color attachments if any is present. If there is no
    color attachment, SAMPLES is equal to the same value from the depth or
    stencil attachment, whichever is present.

    An implementation may only support a subset of the possible
    combinations of sample counts of textures and renderbuffers attached
    to a framebuffer object. The number of supported combinations is
    NUM_SUPPORTED_MULTISAMPLE_MODES_AMD. SUPPORTED_MULTISAMPLE_MODES_AMD
    is an array of NUM_SUPPORTED_MULTISAMPLE_MODES_AMD triples of integers
    where each triple contains a valid combination of sample counts in
    the form {color samples, color storage samples, depth and stencil
    samples}. The first element in each triple is at least 2. The second
    and third element in each triple are at least 1 and are not greater
    than the first element.

    In section 9.2.4, "Renderbuffer Objects", replace the description of
    (Named)RenderbufferStorageMultisample:

    The data storage, format, dimensions, number of samples, and number of
    storage samples of a renderbuffer object’s image are established with
    the commands

      void RenderbufferStorageMultisampleAdvancedAMD( enum target,
          sizei samples, sizei storageSamples, enum internalformat,
          sizei width, sizei height );

      void NamedRenderbufferStorageMultisampleAdvancedAMD(
          uint renderbuffer, sizei samples, sizei storageSamples,
          enum internalformat, sizei width, sizei height );

    For RenderbufferStorageMultisampleAdvancedAMD, the renderbuffer object
    is that bound to <target>, which must be RENDERBUFFER.
    For NamedRenderbufferStorageMultisampleAdvancedAMD, <renderbuffer> is
    the name of the renderbuffer object.

    <internalformat> must be color-renderable, depth-renderable, or
    stencil-renderable (as defined in section 9.4). <width> and <height>
    are the dimensions in pixels of the renderbuffer.

    Upon success, *RenderbufferStorageMultisampleAdvancedAMD deletes any
    existing data store for the renderbuffer image, and the contents of
    the data store are undefined. RENDERBUFFER_WIDTH is set to <width>,
    RENDERBUFFER_HEIGHT is set to <height>, and RENDERBUFFER_INTERNAL_-
    FORMAT is set to <internalformat>.

    If <samples> is zero, then <storageSamples> must be zero, and
    RENDERBUFFER_SAMPLES and RENDERBUFFER_STORAGE_SAMPLES_AMD are set to
    zero. Otherwise <samples> represents a request for a desired minimum
    number of samples and <storageSamples> represents a request for
    a desired minimum number of storage samples, where <storageSamples>
    must not be greater than <samples>. Since different implementations
    may support different sample counts for multisampled rendering,
    the actual number of samples and the actual number of storage samples
    allocated for the renderbuffer image are implementation-dependent.
    However, the resulting value for RENDERBUFFER_SAMPLES is guaranteed
    to be greater than or equal to <samples> and no more than the next
    larger sample count supported by the implementation, and the resulting
    value for RENDERBUFFER_STORAGE_SAMPLES_AMD is guaranteed to be greater
    than or equal to <storageSamples>, no more than the next larger
    storage sample count supported by the implementation, and no more than
    RENDERBUFFER_SAMPLES.

    A GL implementation may vary its allocation of internal component
    resolution based on any *RenderbufferStorageMultisampleAdvancedAMD
    parameter (except <target> and <renderbuffer>), but the allocation and
    chosen internal format must not be a function of any other state and
    cannot be changed once they are established.

    Remove the first 4 errors and add these errors:

    An INVALID_ENUM error is generated by RenderbufferStorageMultisample-
    AdvancedAMD if <target> is not RENDERBUFFER.

    An INVALID_OPERATION error is generated by NamedRenderbufferStorage-
    MultisampleAdvancedAMD if <renderbuffer> is not the name of
    an existing renderbuffer object.

    An INVALID_VALUE error is generated if <samples>, <storageSamples>,
    <width>, or <height> is negative.

    An INVALID_OPERATION error is generated if <internalformat> is a color
    format and <samples> is greater than the implementation-dependent
    limit MAX_COLOR_FRAMEBUFFER_SAMPLES_AMD.

    An INVALID_OPERATION error is generated if <internalformat> is a color
    format and <storageSamples> is greater than the implementation-
    dependent limit MAX_COLOR_FRAMEBUFFER_STORAGE_SAMPLES_AMD.

    An INVALID_OPERATION error is generated if <storageSamples> is greater
    than <samples>.

    An INVALID_OPERATION error is generated if <internalformat> is a depth
    or stencil format and <samples> is greater than the maximum number of
    samples supported for <internalformat> (see GetInternalformativ
    in section 22.3).

    An INVALID_OPERATION error is generated if <internalformat> is a depth
    or stencil format and <storageSamples> is not equal to <samples>.

    Finish the section as follows:

    The commands

        void RenderbufferStorageMultisample( enum target,
            sizei samples, enum internalformat, sizei width,
            sizei height );
        void RenderbufferStorage( enum target, enum internalformat,
            sizei width, sizei height );

    are equivalent to

        RenderbufferStorageMultisampleAdvancedAMD(target, samples,
            samples, internalformat, width, height);

    and

        RenderbufferStorageMultisampleAdvancedAMD(target, 0, 0,
            internalformat, width, height);

    respectively.

    The commands

        void NamedRenderbufferStorageMultisample( uint renderbuffer,
            sizei samples, enum internalformat, sizei width,
            sizei height );
        void NamedRenderbufferStorage( uint renderbuffer,
            enum internalformat, sizei width, sizei height );

    are equivalent to

        NamedRenderbufferStorageMultisampleAdvancedAMD(renderbuffer,
            samples, samples, internalformat, width, height);

    and

        NamedRenderbufferStorageMultisampleAdvancedAMD(renderbuffer,
            0, 0, internalformat, width, height);

    respectively.

    In section 9.2.5, "Required Renderbuffer Formats", replace the last
    paragraph with:

    Implementations must support creation of renderbuffers in these
    required formats with sample counts up to and including:
    * MAX_COLOR_FRAMEBUFFER_SAMPLES_AMD as color renderbuffer samples
    * MAX_COLOR_FRAMEBUFFER_STORAGE_SAMPLES_AMD as color renderbuffer
      storage samples
    * MAX_DEPTH_STENCIL_FRAMEBUFFER_SAMPLES_AMD as depth and stencil
      samples

    with the exception that the signed and unsigned integer formats are
    required only to support creation of renderbuffers with up to
    the value of MAX_INTEGER_SAMPLES samples and storage samples, which
    must be at least one.

    In section 9.2.6, "Renderbuffer Object Queries", replace the paragraph
    mentioning RENDERBUFFER_SAMPLES with:

    If <pname> is RENDERBUFFER_WIDTH, RENDERBUFFER_HEIGHT,
    RENDERBUFFER_INTERNAL_FORMAT, RENDERBUFFER_SAMPLES, or
    RENDERBUFFER_STORAGE_SAMPLES_AMD then <params> will contain the width
    in pixels, height in pixels, internal format, number of samples, or
    number of storage samples, respectively, of the image of
    the renderbuffer object.

    In section 9.4.1, "Framebuffer Attachment Completeness", remove
    the bullet beginning with "If <image> has multisample samples" and
    replace the last 3 bullets about <attachment> with:

    If <attachment> is COLOR_ATTACHMENTi, then <image> must have a color-
    renderable internal format, the sample count must be less than or
    equal to the value of the implementation-dependent limit
    MAX_COLOR_FRAMEBUFFER_SAMPLES_AMD, and the storage sample count must
    be less than or equal to the value of the implementation-dependent
    limit MAX_COLOR_FRAMEBUFFER_STORAGE_SAMPLES_AMD.

    If <attachment> is DEPTH_ATTACHMENT, then <image> must have a depth-
    renderable internal format, and its sample count must be less than or
    equal to the value of the implementation-dependent limit
    MAX_DEPTH_STENCIL_FRAMEBUFFER_SAMPLES_AMD.

    If <attachment> is STENCIL_ATTACHMENT, then <image> must have
    a stencil-renderable internal format, and its sample count must be
    less than or equal to the value of the implementation-dependent limit
    MAX_DEPTH_STENCIL_FRAMEBUFFER_SAMPLES_AMD.

    In section 9.4.2, replace the bullet mentioning RENDERBUFFER_SAMPLES
    with:

    * The value of RENDERBUFFER_SAMPLES of a color attachment defines its
    <number of color samples>; the value of RENDERBUFFER_STORAGE_SAMPLES
    of a color attachment defines its <number of color storage samples>;
    the value of RENDERBUFFER_SAMPLES of a depth or stencil attachment
    defines its <number of depth-stencil samples> for each separately;
    the value of TEXTURE_SAMPLES of a color attachment defines both its
    <number of color samples> and its <number of color storage samples>;
    the value of TEXTURE_SAMPLES of a depth or stencil attachment defines
    its <number of depth-stencil samples> for each separately. If any of
    the defined values is 0, it is treated as 1. Any undefined value is
    treated as equal to any number. For all attachment values that are
    defined, all values of <number of color samples> must be equal, all
    values of <number of color storage samples> must be equal, all values
    of <number of depth-stencil samples> must be equal, and the triple
    {<number of color samples>, <number of color storage samples>, <number
    of depth-stencil samples>} must be in SUPPORTED_MULTISAMPLE_MODES_AMD
    or must be equal to {1, 1, 1}.

    { FRAMEBUFFER_INCOMPLETE_MULTISAMPLE }

Additions to Chapter 17 of the OpenGL 4.5 (Core Profile) Specification
(Writing Fragments and Samples to the Framebuffer)

    In section 17.3.10, "Additional Multisample Fragment Operations", add
    this paragraph after the "If MULTISAMPLE is enabled" paragraph:

    If there are fewer color storage samples (see section 9.2.4) than
    the value of SAMPLES, the number of color storage samples determines
    the number of unique color values that can be stored per pixel.
    The implementation must determine which samples within a pixel share
    the same color value, write that value into 1 color storage sample,
    and remember a mapping between color samples and color storage
    samples to be able to map color storage samples back to color samples.
    The color value equality determination is done in an implementation-
    specific manner, but the implementation must at least recognize a set
    of color samples coming from the same primitive as 1 storage sample if
    sample shading (see section 14.3.1.1) is disabled. If there are not
    enough color storage samples per pixel to store all incoming color
    values, the excessive color values are not stored and the color samples
    with unstored values are marked as having an unknown value. Color
    samples with an unknown value will not contribute to the final color
    value of the pixel when all color samples are resolved by
    BlitFramebuffer (see section 18.3.1).

    If there are fewer depth and stencil samples than the value of SAMPLES
    and implementation-specific optimizations are unable to represent more
    depth and stencil samples within the given storage, the missing depth
    and stencil values should be pulled from or derived from the nearest
    existing depth and stencil samples within the same pixel. The mapping
    from missing to existing depth and stencil samples is implementation-
    specific, but the mapping must be at least:
    * injective if missing samples < existing samples
    * bijective if missing samples = existing samples
    * surjective if missing samples > existing samples
    Depth and stencil tests operate as if the number of depth and stencil
    samples was equal to the value of SAMPLES.

Errors

    An INVALID_ENUM error is generated by RenderbufferStorageMultisample-
    AdvancedAMD if <target> is not RENDERBUFFER.

    An INVALID_OPERATION error is generated by NamedRenderbufferStorage-
    MultisampleAdvancedAMD if <renderbuffer> is not the name of
    an existing renderbuffer object.

    An INVALID_VALUE error is generated if <samples>, <storageSamples>,
    <width>, or <height> is negative.

    An INVALID_OPERATION error is generated if <internalformat> is a color
    format and <samples> is greater than the implementation-dependent
    limit MAX_COLOR_FRAMEBUFFER_SAMPLES_AMD.

    An INVALID_OPERATION error is generated if <internalformat> is a color
    format and <storageSamples> is greater than the implementation-
    dependent limit MAX_COLOR_FRAMEBUFFER_STORAGE_SAMPLES_AMD.

    An INVALID_OPERATION error is generated if <storageSamples> is greater
    than <samples>.

    An INVALID_OPERATION error is generated if <internalformat> is a depth
    or stencil format and <samples> is greater than the maximum number of
    samples supported for <internalformat> (see GetInternalformativ
    in section 22.3).

    An INVALID_OPERATION error is generated if <internalformat> is a depth
    or stencil format and <storageSamples> is not equal to <samples>.

New State

    Add to Table 23.27, "Renderbuffer (state per renderbuffer object)"
                                                                        Initial
    Get Value                         Type  Get Command                 Value    Description             Section
    --------------------------------  ----  --------------------------  -------  ----------------------  -------
    RENDERBUFFER_STORAGE_SAMPLES_AMD   Z+   GetRenderbufferParameteriv  0        No. of storage samples  9.2.4

New Implementation Dependent Values
                                                                       Minimum
    Get Value                                 Type        Get Command  Value    Description                             Section
    ----------------------------------------  ----------  -----------  -------  ---------------------------------------  -------
    MAX_COLOR_FRAMEBUFFER_SAMPLES_AMD             Z+      GetIntegerv  4        Max. no. of color samples supported by   9.2.4
                                                                                framebuffer objects.
    MAX_COLOR_FRAMEBUFFER_STORAGE_SAMPLES_AMD     Z+      GetIntegerv  4        Max. no. of color storage samples        9.2.4
                                                                                supported by framebuffer objects.
    MAX_DEPTH_STENCIL_FRAMEBUFFER_SAMPLES_AMD     Z+      GetIntegerv  4        Max. no. of depth and stencil samples    9.2.4
                                                                                supported by framebuffer objects.
    NUM_SUPPORTED_MULTISAMPLE_MODES_AMD           Z+      GetIntegerv  1        No. of supported combinations of color   9.2.4
                                                                                samples, color storage samples, and
                                                                                depth-stencil samples by framebuffer
                                                                                objects.
    SUPPORTED_MULTISAMPLE_MODES_AMD           n * 3 x Z+  GetIntegerv  -        NUM_SUPPORTED_MULTISAMPLE_MODES_AMD (n)  9.2.4
                                                                                triples of integers. Each triple is
                                                                                a unique combination of color samples,
                                                                                color storage samples, and depth-stencil
                                                                                samples supported by framebuffer objects.

AMD Implementation Details

    The following multisample modes are supported by AMD's open source
    OpenGL driver:

                 Color    Depth &
        Color    storage  stencil
        samples  samples  samples
        =======  =======  =======
        16       8        8
        16       4        8
        16       2        8
        16       4        4
        16       2        4
        16       2        2
        -------  -------  -------
        8        8        8
        8        4        8
        8        2        8
        8        4        4
        8        2        4
        8        2        2
        -------  -------  -------
        4        4        4
        4        2        4
        4        2        2
        -------  -------  -------
        2        2        2

Issues

    None.

Revision History

    Rev.    Date    Author    Changes
    ----  --------  --------  --------------------------------------------
     1    06/28/18  mareko    Initial version
