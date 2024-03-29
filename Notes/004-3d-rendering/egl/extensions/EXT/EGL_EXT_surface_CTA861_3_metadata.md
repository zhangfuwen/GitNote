# EXT_surface_CTA861_3_metadata

Name

    EXT_surface_CTA861_3_metadata

Name Strings

    EGL_EXT_surface_CTA861_3_metadata

Contact

    Courtney Goeltzenleuchter (courtneygo 'at' google.com)

Contributors

    James Jones
    Weiwan Liu

IP Status

    No known IP claims.

Status

    Complete

Version

    Version 5 - Nov 29, 2016

Number

    EGL Extension #117

Dependencies

    This extension requires EGL 1.5 and EGL_EXT_surface_SMPTE2086_metadata.

    This extension is written against the wording of the EGL 1.5 specification
    (August 27, 2014).

Overview

    This extension adds additional EGL surface attributes for the metadata
    defined by the CTA (Consumer Technology Association) 861.3 standard.
    This metadata, in addition to the SMPTE 2086 metadata, is used to define the
    color volume of the mastering display as well as the content (CTA-861.3),
    The mastering display is the display where creative work is done and creative
    intent is established. To preserve such creative intent as much as possible
    and achieve consistent color reproduction on different viewing displays,
    it is useful for the display pipeline to know the color volume of the
    original mastering display where content is created or tuned.  This avoids
    performing unnecessary mapping of colors that are not displayable on the
    original mastering display.

    This extension adds the ability to pass the CTA-861.3 metadata via EGL,
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

	EGL_CTA861_3_MAX_CONTENT_LIGHT_LEVEL_EXT            0x3360
	EGL_CTA861_3_MAX_FRAME_AVERAGE_LEVEL_EXT            0x3361

Additions to Chapter "3.5.6 Surface Attributes" of the EGL 1.5 Specification

    Add the following paragraph before the "Errors" section on page 43,

        If attribute is EGL_CTA861_3_MAX_CONTENT_LIGHT_LEVEL_EXT or EGL_CTA861_3_-
	MAX_FRAME_AVERAGE_LEVEL_EXT, then value indicates the corresponding
	maximum content light level and maximum frame average level.
        The unit of value is 1 nit (candela per square meter). The
        floating-point luminance values should be multiplied by
        EGL_METADATA_SCALING, a constant scaling factor of 50000, before being
        passed into eglSurfaceAttrib as integers.

        Exactly how the color volume information is used to assist the color
	reproduction process is implementation dependant.

        The initial values of EGL_CTA861_3_MAX_CONTENT_LIGHT_LEVEL_EXT and
	EGL_CTA861_3_MAX_FRAME_AVERAGE_LEVEL_EXT are EGL_DONT_CARE, which causes the
        hints to be ignored. If value is not in the implementation's supported
        range for attribute, a EGL_INVALID_VALUE error is generated, and some or
        all of the metadata fields are ignored.

    Change the original footnote 12 at the end of section "3.5.6 Surface
    Attributes" on page 45 to,

            [13] EGL_DISPLAY_SCALING (10000) and EGL_METADATA_SCALING_EXT (50000)
        are used where EGL needs to take or return floating-point attribute
        values, which would normally be smaller than 1, as integers while still
        retaining sufficient precision to be meaningful.

    Addition to Table 3.5 "Queryable surface attributes and types",

                      Attribute                   Type
        ------------------------------------------------
	EGL_CTA861_3_MAX_CONTENT_LIGHT_LEVEL_EXT     integer
	EGL_CTA861_3_MAX_FRAME_AVERAGE_LEVEL_EXT     integer

                      Description
        --------------------------------------------------------------------------------------
        Maximum content light level in nit multiplied by EGL_METADATA_SCALING_EXT
        Maximum frame average light level in nit multiplied by EGL_METADATA_SCALING_EXT

    Add the following paragraph at the end of section "3.5.6 Surface Attributes"
    on page 45,

        Querying EGL_CTA861_3_MAX_CONTENT_LIGHT_LEVEL_EXT EGL_CTA861_3_MAX_-
	FRAME_AVERAGE_LEVEL_EXT returns respectively the maximum content light level
	and maximum frame average level respectively. The values returned are
	in units of 1 nit (candela per square meter), multiplied by the constant
	value EGL_METADATA_SCALING_EXT (50000). The value of EGL_CTA861_3_MAX_-
	CONTENT_LIGHT_LEVEL_EXT and EGL_CTA861_3_MAX_FRAME_AVERAGE_LEVEL_EXT can
	be set via eglSurfaceAttrib as described above.

Errors

    Described in the body text above.

Issues

Revision History

    Version 1, 2017/02/28
      - Initial draft

