# EXT_surface_SMPTE2086_metadata

Name

    EXT_surface_SMPTE2086_metadata

Name Strings

    EGL_EXT_surface_SMPTE2086_metadata

Contact

    Weiwan Liu (weiwliu 'at' nvidia.com)

Contributors

    Tom Cooksey
    Courtney Goeltzenleuchter
    Mathias Heyer
    Lauri Hyvarinen
    James Jones
    Daniel Koch
    Jeff Leger
    Sandeep Shinde

IP Status

    No known IP claims.

Status

    Complete

Version

    Version 8 - Oct 16, 2017

Number

    EGL Extension #109

Dependencies

    This extension requires EGL 1.4.

    This extension is written against the wording of the EGL 1.5 specification
    (August 27, 2014).

Overview

    This extension adds a new set of EGL surface attributes for the metadata
    defined by the SMPTE (the Society of Motion Picture and Television
    Engineers) ST 2086 standard. The SMPTE 2086 metadata includes the color
    primaries, white point and luminance range of the mastering display, which
    all together define the color volume that contains all the possible colors
    the mastering display can produce. The mastering display is the display
    where creative work is done and creative intent is established. To preserve
    such creative intent as much as possible and achieve consistent color
    reproduction on different viewing displays, it is useful for the display
    pipeline to know the color volume of the original mastering display where
    content is created or tuned. This avoids performing unnecessary mapping of
    colors that are not displayable on the original mastering display.

    This extension adds the ability to pass the SMPTE 2086 metadata via EGL,
    from which the color volume can be derived. While the general purpose of the
    metadata is to assist in the transformation between different color volumes
    of different displays and help achieve better color reproduction, it is not
    in the scope of this extension to define how exactly the metadata should be
    used in such a process. It is up to the implementation to determine how to
    make use of the metadata.

New Procedures and Functions

    None.

New Tokens

    Accepted as attribute by eglSurfaceAttrib and eglQuerySurface:

        EGL_SMPTE2086_DISPLAY_PRIMARY_RX_EXT       0x3341
        EGL_SMPTE2086_DISPLAY_PRIMARY_RY_EXT       0x3342
        EGL_SMPTE2086_DISPLAY_PRIMARY_GX_EXT       0x3343
        EGL_SMPTE2086_DISPLAY_PRIMARY_GY_EXT       0x3344
        EGL_SMPTE2086_DISPLAY_PRIMARY_BX_EXT       0x3345
        EGL_SMPTE2086_DISPLAY_PRIMARY_BY_EXT       0x3346
        EGL_SMPTE2086_WHITE_POINT_X_EXT            0x3347
        EGL_SMPTE2086_WHITE_POINT_Y_EXT            0x3348
        EGL_SMPTE2086_MAX_LUMINANCE_EXT            0x3349
        EGL_SMPTE2086_MIN_LUMINANCE_EXT            0x334A

Additions to Chapter "3.5.6 Surface Attributes" of the EGL 1.5 Specification

    Add the following paragraph before the "Errors" section on page 43,

        If attribute is EGL_SMPTE2086_DISPLAY_PRIMARY_RX_EXT, EGL_SMPTE2086_-
        DISPLAY_PRIMARY_RY_EXT, EGL_SMPTE2086_DISPLAY_PRIMARY_GX_EXT, EGL_-
        SMPTE2086_DISPLAY_PRIMARY_GY_EXT, EGL_SMPTE2086_DISPLAY_PRIMARY_BX_EXT
        or EGL_SMPTE2086_DISPLAY_PRIMARY_BY_EXT, then value indicates the
        corresponding xy chromaticity coordinate[12] of the mastering display's
        red, green or blue color primary, as configured for the mastering
        process. The floating-point display primary coordinates should be
        multiplied by EGL_METADATA_SCALING_EXT (50000)[13], before being passed
        into eglSurfaceAttrib as integers.

        If attribute is EGL_SMPTE2086_WHITE_POINT_X_EXT or EGL_SMPTE2086_WHITE_-
        POINT_Y_EXT, then value indicates the corresponding xy chromaticity
        coordinate[12] of the mastering display's white point, as configured for
        the mastering process. The floating-point white point chromaticity
        coordinates should be multiplied by EGL_METADATA_SCALING_EXT (50000),
        before being passed into eglSurfaceAttrib as integers.

        If attribute is EGL_SMPTE2086_MAX_LUMINANCE_EXT or EGL_SMPTE2086_MIN_-
        LUMINANCE_EXT, then value indicates the maximum or minimum display
        luminance of the mastering display, as configured for the mastering
        process. The unit of value is 1 nit (candela per square meter). The
        floating-point luminance values should be multiplied by
        EGL_METADATA_SCALING_EXT, a constant scaling factor of 50000, before
        being passed into eglSurfaceAttrib as integers.

        By defining the mastering display's color volume through color
        primaries, white point, and luminance range, applications give EGL
        and the underlying display pipeline hints as to how to reproduce colors
        more closely to the original content when created on the mastering
        display. Exactly how the color volume information is used to assist the
        color reproduction process is implementation dependant.

        The initial values of EGL_SMPTE2086_DISPLAY_PRIMARY_RX_EXT, EGL_-
        SMPTE2086_DISPLAY_PRIMARY_RY_EXT, EGL_SMPTE2086_DISPLAY_PRIMARY_GX_EXT,
        EGL_SMPTE2086_DISPLAY_PRIMARY_GY_EXT, EGL_SMPTE2086_DISPLAY_PRIMARY_BX_-
        EXT, EGL_SMPTE2086_DISPLAY_PRIMARY_BY_EXT, EGL_SMPTE2086_WHITE_POINT_X_-
        EXT, EGL_SMPTE2086_WHITE_POINT_Y_EXT, EGL_SMPTE2086_MAX_LUMINANCE_EXT
        and EGL_SMPTE2086_MIN_LUMINANCE_EXT are EGL_DONT_CARE, which causes the
        hints to be ignored. If value is not in the implementation's supported
        range for attribute, a EGL_BAD_PARAMETER error is generated, and some or
        all of the metadata fields are ignored.

    Add the following footnote at the end of page 43, and increment all the
    subsequent footnote numbers in Chapter 3,

            [12] Chromaticity coordinates x and y are as specified in CIE
        15:2004 "Calculation of chromaticity coordinates" (Section 7.3) and are
        limited to between 0 and 1 for real colors for the mastering display.

    Change the original footnote 12 at the end of section "3.5.6 Surface
    Attributes" on page 45 to,

            [13] EGL_DISPLAY_SCALING (10000) and EGL_METADATA_SCALING_EXT (50000)
        are used where EGL needs to take or return floating-point attribute
        values, which would normally be smaller than 1, as integers while still
        retaining sufficient precision to be meaningful.

    Addition to Table 3.5 "Queryable surface attributes and types",

                      Attribute                   Type
        ------------------------------------------------
        EGL_SMPTE2086_DISPLAY_PRIMARY_RX_EXT     integer
        EGL_SMPTE2086_DISPLAY_PRIMARY_RY_EXT     integer
        EGL_SMPTE2086_DISPLAY_PRIMARY_GX_EXT     integer
        EGL_SMPTE2086_DISPLAY_PRIMARY_GY_EXT     integer
        EGL_SMPTE2086_DISPLAY_PRIMARY_BX_EXT     integer
        EGL_SMPTE2086_DISPLAY_PRIMARY_BY_EXT     integer
        EGL_SMPTE2086_WHITE_POINT_X_EXT          integer
        EGL_SMPTE2086_WHITE_POINT_Y_EXT          integer
        EGL_SMPTE2086_MAX_LUMINANCE_EXT          integer
        EGL_SMPTE2086_MIN_LUMINANCE_EXT          integer

                      Description
        ------------------------------------------------------------------------------------------
        x chromaticity coordinate for red display primary multiplied by EGL_METADATA_SCALING_EXT
        y chromaticity coordinate for red display primary multiplied by EGL_METADATA_SCALING_EXT
        x chromaticity coordinate for green display primary multiplied by EGL_METADATA_SCALING_EXT
        y chromaticity coordinate for green display primary multiplied by EGL_METADATA_SCALING_EXT
        x chromaticity coordinate for blue display primary multiplied by EGL_METADATA_SCALING_EXT
        y chromaticity coordinate for blue display primary multiplied by EGL_METADATA_SCALING_EXT
        x chromaticity coordinate for white point multiplied by EGL_METADATA_SCALING_EXT
        y chromaticity coordinate for white point multiplied by EGL_METADATA_SCALING_EXT
        Maximum luminance in nit multiplied by EGL_METADATA_SCALING_EXT
        Minimum luminance in nit multiplied by EGL_METADATA_SCALING_EXT

    Add the following paragraph at the end of section "3.5.6 Surface Attributes"
    on page 45,

        Querying EGL_SMPTE2086_DISPLAY_PRIMARY_RX_EXT, EGL_SMPTE2086_DISPLAY_-
        PRIMARY_RY_EXT, EGL_SMPTE2086_DISPLAY_PRIMARY_GX_EXT, EGL_SMPTE2086_-
        DISPLAY_PRIMARY_GY_EXT, EGL_SMPTE2086_DISPLAY_PRIMARY_BX_EXT or EGL_-
        SMPTE2086_DISPLAY_PRIMARY_BY_EXT returns respectively the xy
        chromaticity coordinate of the mastering display's red, green or blue
        color primary, multiplied by the constant value EGL_METADATA_SCALING_EXT
        (50000). The display primary coordinates can be set via eglSurfaceAttrib
        as described above.

        Querying EGL_SMPTE2086_WHITE_POINT_X_EXT, or EGL_SMPTE2086_WHITE_POINT_-
        Y_EXT returns respectively the xy chromaticity coordinate of the
        mastering display's white point, multiplied by the constant value EGL_-
        METADATA_SCALING (50000). The white point coordinates can be set via
        eglSurfaceAttrib as described above.

        Querying EGL_SMPTE2086_MAX_LUMINANCE_EXT or EGL_SMPTE2086_MIN_-
        LUMINANCE_EXT returns respectively the maximum and minimum display
        luminance of the mastering display. The values returned are in units of
        1 nit (candela per square meter), multiplied by the constant value EGL_-
        METADATA_SCALING (50000). The value of EGL_SMPTE2086_MAX_LUMINANCE_EXT
        and EGL_SMPTE2086_MIN_LUMINANCE_EXT can be set via eglSurfaceAttrib as
        described above.

Errors

    Described in the body text above.

Issues

    1. Should this extension define a valid data range for each metadata field?

       RESOLVED: No. It is not in the scope of this extension to define how the
       metadata hints should be used in the display pipeline and, as a result,
       what the valid data ranges are for the metadata fields. It is
       implementation dependant, but related standards, such as SMPTE ST 2086,
       can be used as reference. As described in the body, implemetations may
       generate a EGL_BAD_PARAMTER error to notify applications that the input
       metadata values are invalid or not supported.

Revision History

    Version 1, 2016/04/22
      - Initial draft

    Version 2, 2016/05/25
      - Rename to EXT and introduce a new scaling factor

    Version 3, 2016/10/19
      - Add an error and revise issue 1

    Version 4, 2016/11/22
      - Change status to complete

    Version 5, 2016/11/29
      - Add token assigments

    Version 6, 2017/02/28
      - Add 'EXT' suffix to 'EGL_METADATA_SCALING'

    Version 7, 2017/10/13
      - Rename EGL_INVALID_VALUE (which doesn't exist) to EGL_FALSE

    Version 8, 2017/10/16
      - Fix v7 change to use EGL_BAD_PARAMETER as the error code
	generated vs. EGL_FALSE which is the expected return value of
	the function.

