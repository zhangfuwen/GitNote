# MESA_tile_raster_order

Name

    MESA_tile_raster_order

Name Strings

    GL_MESA_tile_raster_order

Contact

    Eric Anholt, Broadcom (eric@anholt.net)

Status

    Proposal

Version

    Last modified date: 24 September 2017

Number

    OpenGL Extension #515
    OpenGL ES Extension #292

Dependencies

    GL_ARB_texture_barrier or GL_NV_texture_barrier is required.

    This extension is written against the OpenGL 4.4 (Compatibility
    Profile) specification, as modified by the GL_ARB_texture_barrier
    extension.

Overview

    This extension extends the sampling-from-the-framebuffer behavior provided
    by GL_ARB_texture_barrier to allow setting the rasterization order of the
    scene, so that overlapping blits can be implemented.  This can be used for
    scrolling or window movement within in 2D scenes, without first copying to
    a temporary.

IP Status

    None

Issues

    1.  Should this extension also affect BlitFramebuffer?

        NOT RESOLVED: BlitFramebuffer could use the same underlying
        functionality to provide defined results for 1:1 overlapping blits,
        but one could use the coordinates being copied to just produce the
        right result automatically, rather than requiring the state flags to
        be adjusted.

New Procedures and Functions

    None

New Tokens

    None

Additions to Chapter 9 of the OpenGL 4.4 Specification (Per-Fragment
Operations and the Frame Buffer)

    Modify Section 9.3.1 Rendering Feedback Loops, p. 289

    Replace the bullet point "If a texel has been written..." with:

      - A texel has been written, but it has been separated from this
        Draw call by the command:

          void TextureBarrier(void);

        TextureBarrier() will guarantee that writes have completed and
        caches have been invalidated before subsequent Draws are
        executed."

      - TILE_RASTER_ORDER_FIXED_MESA is enabled, and there is only a
        single write of each texel, and primitives are emitted in the
        order of TILE_RASTER_ORDER_INCREASING_X/Y_MESA (where those
        being disabled mean negative texel offsets), and reads are
        only performed from texels offset from the current fragment
        shader invocation in the direction specified by
        TILE_RASTER_ORDER_INCREASING_X/Y_MESA, e.g. using
        "texelFetch2D(sampler, ivec2(gl_FragCoord.xy + vec2(dx, dy)),
        0);".

Additions to the AGL/GLX/WGL Specifications

    None

GLX Protocol

    None

Errors

    None

New State

    Get Value                            Type    Get Command    Initial Value Description                  Section   Attribute
    -----------------------------------  ------  -------------  ------------- ---------------------------  --------  ------------
    TILE_RASTER_ORDER_FIXED_MESA         B       IsEnabled      True          Tile rasterization order is  9.3.1     enable
                                                                              defined by
                                                                              TILE_RASTER_ORDER_INCREASING_*_MESA.
                                                                              in increasing X direction
    TILE_RASTER_ORDER_INCREASING_X_MESA  B       IsEnabled      True          Tiles are rasterized         9.3.1     enable
                                                                              in increasing X direction
    TILE_RASTER_ORDER_INCREASING_Y_MESA  B       IsEnabled      True          Tiles are rasterized         9.3.1     enable
                                                                              in increasing Y direction

Revision History

    26 July 2017 - Initial draft

    24 September 2017 - Improved wording of the new specification
                        paragraph.  Adjust the ARB_texture_barrier
                        paragraph, to make it clear that it's not
                        required when in tile raster order mode.

    2 October 2017 - Give it an ES extension number, mention
                     NV_texture_barrier for ES.

    5 October 2017 - Mention what spec it's written against.
