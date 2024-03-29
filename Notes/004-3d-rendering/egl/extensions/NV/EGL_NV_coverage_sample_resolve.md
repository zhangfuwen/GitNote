# NV_coverage_sample_resolve

Name

    NV_coverage_sample_resolve

Name Strings

    EGL_NV_coverage_sample_resolve

Contact

    James Jones, NVIDIA Corporation (jajones 'at' nvidia.com)

Notice

    Copyright NVIDIA Corporation, 2011

Status

    NVIDIA Proprietary

Version

    Last Modified Date:  2011/04/13
    NVIDIA Revision: 1.0

Number

    EGL Extension #30

Dependencies

    Written based on the wording of the EGL 1.4 specification.

    Trivially interacts with EGL_NV_coverage_sample

    Requires EGL 1.2.

Overview

    NV_coverage_sample introduced a method to improve rendering quality
    using a separate buffer to store coverage information for pixels in
    the color buffers.  It also provided a mechanism to disable writing
    to the coverage buffer when coverage sample filtering was not needed
    or undesirable.  However, it did not provide a way to disable
    reading data from the coverage buffer at resolve time.  In some
    cases performance can be improved by eliminating these memory reads.
    To that end, this extension exposes a surface attribute that allows
    applications to specify when no coverage sample resolve is desired.

IP Status

    NVIDIA Proprietary

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted by the <attribute> parameter of eglSurfaceAttrib and
    eglQuerySurface:

        EGL_COVERAGE_SAMPLE_RESOLVE_NV              0x3131

    Accepted by the <value> parameter of eglSurfaceAttrib and returned
    in the <value> parameter of eglQuerySurface when <attribute> is
    EGL_COVERAGE_SAMPLE_RESOLVE_NV:

        EGL_COVERAGE_SAMPLE_RESOLVE_DEFAULT_NV      0x3132
        EGL_COVERAGE_SAMPLE_RESOLVE_NONE_NV         0x3133

Additions to Chapter 3 of the EGL 1.4 Specification (EGL Functions and
Errors)

    Additions to section 3.5.6 (Surface Attributes)

    Replace the last sentence of paragraph 2 (p. 35):

    "Attributes that can be specified are
    EGL_COVERAGE_SAMPLE_RESOLVE_NV, EGL_MIPMAP_LEVEL,
    EGL_MULTISAMPLE_RESOLVE, and EGL_SWAP_BEHAVIOR."

    Add the following paragraphs between paragraphs 2 and 3 (p. 35):

    "If <attribute> is EGL_COVERAGE_SAMPLE_RESOLVE_NV, then <value>
    specifies the filter to use when resolving the coverage sample
    buffer.  A <value> of EGL_COVERAGE_SAMPLE_RESOLVE_DEFAULT_NV chooses
    the default implementation-defined filtering method, while
    EGL_MULTISAMPLE_RESOLVE_NONE_NV disables filtering based on coverage
    data.

    "The initial value of EGL_COVERAGE_SAMPLE_RESOLVE_NV is
    EGL_COVERAGE_SAMPLE_RESOLVE_DEFAULT_NV."

    Add the following paragraph after paragraph 13 (p. 36):

    "Querying EGL_COVERAGE_SAMPLE_RESOLVE_NV returns the filtering
    method used when performing coverage buffer resolution.  The filter
    may be either EGL_COVERAGE_SAMPLE_RESOLVE_DEFAULT_NV or
    EGL_COVERAGE_SAMPLE_RESOLVE_NONE_NV, as described above for
    eglSurfaceAttrib."

Interactions with EGL_NV_coverage_sample:

    This extension relies on language in EGL_NV_coverage_sample to
    describe the coverage sample buffer.

    If EGL_NV_coverage_sample is not present, this extension has no
    effect on rendering.

Issues

    1.  Should it be an error to set EGL_COVERAGE_SAMPLE_RESOLVE_NV on
        surfaces that don't have a coverage buffer?

        RESOLVED:  No.  EGL_COVERAGE_SAMPLE_RESOLVE_DEFAULT_NV will behave
        the same as EGL_COVERAGE_SAMPLE_RESOLVE_NONE_NV in this case.

Revision History

#1  (James Jones, 2011-04-13)

    - Initial revision.
