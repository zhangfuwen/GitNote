# OES_point_sprite

Name

    OES_point_sprite

Name Strings

    GL_OES_point_sprite

Contact

    Aaftab Munshi (amunshi@ati.com)

Notice

    Copyright (c) 2004-2013 The Khronos Group Inc. Copyright terms at
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

    Ratified by the Khronos BOP, Aug 5, 2004.

IP Status

    No known IP issues.

Version

    Last Modified Date: August 5, 2004

Number

    OpenGL ES Extension #15

Dependencies

    OpenGL ES 1.0 is required

Overview

    Applications such as particle systems have tended to use OpenGL quads
    rather than points to render their geometry, since they would like
    to use a custom-drawn texture for each particle, rather than the
    traditional OpenGL round antialiased points, and each fragment in
    a point has the same texture coordinates as every other fragment.

    Unfortunately, specifying the geometry for these quads can be
    expensive, since it quadruples the amount of geometry required, and
    may also require the application to do extra processing to compute
    the location of each vertex.

    The purpose of this extension is to allow such applications to use
    points rather than quads.  When GL_POINT_SPRITE_OES is enabled,
    the state of point antialiasing is ignored.  For each texture unit,
    the app can then specify whether to replace the existing texture
    coordinates with point sprite texture coordinates, which are
    interpolated across the point.

Issues

    The following are the list of issues as discussed in the
    ARB_point_sprite extension.  I've deleted issues that are not related
    to OpenGL ES such as vertex shader programs etc.

    Tokens that use _ARB names are modified to use _OES.

    *   Should this spec say that point sprites get converted into quads?

        RESOLVED: No, this would make the spec much uglier, because then
        we'd have to say that polygon smooth and stipple get turned off,
        etc.  Better to provide a formula for computing the texture
        coordinates and leave them as points.

    *   How are point sprite texture coordinates computed?

        RESOLVED: They move smoothly as the point moves around on the
        screen, even though the pixels touched by the point do not.  The
        exact formula is given in the spec below.

        A point sprite can be thought of as a quad whose upper-left corner has
        (s,t) texture coordinates of (0,0) and whose lower-right corner has
        texture coordinates of (1,1), as illustrated in the following figure.
        In the figure "P" is the center of the point sprite, and "O" is the
        origin (0,0) of the window coordinate system.  Note that the y window
        coordinate increases from bottom-to-top but the t texture coordinate
        of point sprites increases from top-to-bottom.

              ^
            +y| (0,0)
              |   +-----+
              |   |     |
              |   |  P  |
              |   |     |
              |   +-----+
              |       (1,1)
              |              +x
              O--------------->

        Applications using a single texture for both point sprites and other
        geometry need to account for the fixed coordinate mapping of point
        sprites.

    *   How do point sizes for point sprites work?

        RESOLVED: This specification treats point sprite sizes like
        antialiased point sizes, but with more leniency.  Implementations
        may choose to not clamp the point size to the antialiased point
        size range.  The set of point sprite sizes available must be
        a superset of the antialiased point sizes.  However, whereas
        antialiased point sizes are all evenly spaced by the point size
        granularity, point sprites can have an arbitrary set of sizes.
        This lets implementations use, e.g., floating-point sizes.

    *   Should there be a way to query the list of supported point sprite
        sizes?

        RESOLVED: No.  If an implementation were to use, say, a single-
        precision IEEE float to represent point sizes, the list would be
        rather long.

    *   Do mipmaps apply to point sprites?

        RESOLVED: Yes.  They are similar to quads in this respect.

    *   What of this extension's state is per-texture unit and what
        of this extension's state is state is global?

        RESOLVED: The GL_POINT_SPRITE_OES enable is global.  The
        COORD_REPLACE_OES state is per-texture unit (state set by TexEnv is
        per-texture unit).

    *   Should there be a global on/off switch for point sprites, or
        should the per-unit enable imply that switch?

        RESOLVED: There is a global switch to turn it on and off.  This
        is probably more convenient for both driver and app, and it
        simplifies the spec.

    *   What should the TexEnv mode for point sprites be called?

        RESOLVED: COORD_REPLACE_OES.

    *   What is the interaction with multisample points, which are round?

        RESOLVED: Point sprites are rasterized as squares, even in
        multisample mode.  Leaving them as round points would make the
        feature useless.

    *   How does this extension interact with the point size attenuation
        functionality in OES_point_parameters and OpenGL 1.4?

        RESOLVED:  Point sprites sizes are attenuated just like the sizes of
        non-sprite points.

    *   How are point sprites clipped?

        RESOLVED:  Point sprites are transformed as points, and standard point
        clipping operations are performed.  This can cause point sprites that
        move off the edge of the screen to disappear abruptly, in the same way
        that regular points do.  As with any other primitive, standard
        per-fragment clipping operations (scissoring, window ownership test)
        still apply.

New Procedures and Functions

    None

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, and by the
    <target> parameter of TexEnvf, TexEnvfv, TexEnvx, TexEnvxv:

        POINT_SPRITE_OES                               0x8861

    When the <target> parameter of TexEnvf, TexEnvfv, TexEnvx, TexEnvxv,
    is POINT_SPRITE_OES, then the value of <pname> may be:

        COORD_REPLACE_OES                              0x8862

    When the <target> and <pname> parameters of TexEnvf, TexEnvfv,
    TexEnvx, TexEnvxv, are POINT_SPRITE_OES and COORD_REPLACE_OES
    respectively, then the value of <param> or the value pointed
    to by <params> may be:

        FALSE
        TRUE


Additions to Chapter 2 of the OpenGL 1.4 Specification (OpenGL Operation)

    None.

Additions to Chapter 3 of the OpenGL 1.4 Specification (Rasterization)

    Insert the following paragraphs after the second paragraph of section
    3.3 (page 66):

    "Point sprites are enabled or disabled by calling Enable or Disable
    with the symbolic constant POINT_SPRITE_OES.  The default state is for
    point sprites to be disabled.  When point sprites are enabled, the
    state of the point antialiasing enable is ignored.

    The point sprite texture coordinate replacement mode is set with one
    of the commands

      void TexEnv{ixf}(enum target, enum pname, T param)
      void TexEnv{ixf}v(enum target, enum pname, const T *params)

    where target is POINT_SPRITE_OES and pname is COORD_REPLACE_OES.  The
    possible values for param are FALSE and TRUE.  The default value for
    each texture unit is for point sprite texture coordinate replacement
    to be disabled."

    Replace the first two sentences of the second paragraph of section
    3.3.1 (page 67) with the following:

    "The effect of a point width other than 1.0 depends on the state of
    point antialiasing and point sprites.  If antialiasing and point
    sprites are disabled, ..."

    Replace the first sentences of the fourth paragraph of section 3.3.1
    (page 68) with the following:

    "If antialiasing is enabled and point sprites are disabled, ..."

    Insert the following paragraphs at the end of section 3.3.1 (page
    70):

    "When point sprites are enabled, then point rasterization produces a
    fragment for each framebuffer pixel whose center lies inside a square
    centered at the point's (x_w, y_w), with side length equal to the
    current point size.

    All fragments produced in rasterizing a point sprite are assigned the
    same associated data, which are those of the vertex corresponding to
    the point, with texture coordinates s, t, and r replaced with s/q,
    t/q, and r/q, respectively.  If q is less than or equal to zero, the
    results are undefined.  However, for each texture unit where
    COORD_REPLACE_OES is TRUE, these texture coordinates are replaced
    with point sprite texture coordinates.  The s coordinate varies
    from 0 to 1 across the point horizontally left-to-right, while
    the t coordinate varies from 0 to 1 vertically top-to-bottom.
    The r and q coordinates are replaced with the constants 0 and 1,
    respectively.

    The following formula is used to evaluate the s and t coordinates:

      s = 1/2 + (x_f + 1/2 - x_w) / size
      t = 1/2 - (y_f + 1/2 - y_w) / size

    where size is the point's size, x_f and y_f are the (integral) window
    coordinates of the fragment, and x_w and y_w are the exact, unrounded
    window coordinates of the vertex for the point.

    The widths supported for point sprites must be a superset of those
    supported for antialiased points.  There is no requirement that these
    widths must be equally spaced.  If an unsupported width is requested,
    the nearest supported width is used instead."

    Replace the text of section 3.3.2 (page 70) with the following:

    "The state required to control point rasterization consists of the
    floating-point point width, three floating-point values specifying
    the minimum and maximum point size and the point fade threshold size,
    three floating-point values specifying the distance attenuation
    coefficients, a bit indicating whether or not antialiasing is
    enabled, a bit indicating whether or not point sprites are enabled,
    and a bit for the point sprite texture coordinate replacement mode
    for each texture unit."

    Replace the text of section 3.3.3 (page 70) with the following:

    "If MULTISAMPLE is enabled, and the value of SAMPLE_BUFFERS is one,
    then points are rasterized using the following algorithm, regardless
    of whether point antialiasing (POINT_SMOOTH) is enabled or disabled.
    Point rasterization produces a fragment for each framebuffer pixel
    with one or more sample points that intersect a region centered at
    the point's (x_w, y_w).  This region is a circle having diameter
    equal to the current point width if POINT_SPRITE_OES is disabled, or
    a square with side equal to the current point width if
    POINT_SPRITE_OES is enabled.  Coverage bits that correspond to sample
    points that intersect the region are 1, other coverage bits are 0.
    All data associated with each sample for the fragment are the data
    associated with the point being rasterized, with the exception of
    texture coordinates when POINT_SPRITE_OES is enabled; these texture
    coordinates are computed as described in section 3.3.

    Point size range and number of gradations are equivalent to those
    supported for antialiased points when POINT_SPRITE_OES is disabled.
    The set of point sizes supported is equivalent to those for point
    sprites without multisample when POINT_SPRITE_OES is enabled."

Additions to Chapter 4 of the OpenGL 1.4 Specification (Per-Fragment
Operations and the Frame Buffer)

    None.

Additions to Chapter 5 of the OpenGL 1.4 Specification (Special
Functions)

    None.

Additions to Chapter 6 of the OpenGL 1.4 Specification (State and
State Requests)

    None.

Errors

    None.

New State

(table 6.12, p. 220)

Get Value                Type    Get Command     Initial Value   Description
---------                ----    -----------     -------------   -----------
POINT_SPRITE_OES         B       IsEnabled       False           point sprite enable

(table 6.17, p. 225)

Get Value                Type    Get Command     Initial Value   Description
---------                ----    -----------     -------------   -----------
COORD_REPLACE_OES        2* x B  GetTexEnviv     False           coordinate replacement
                                                                 enable

Revision History


