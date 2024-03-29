# NV_conservative_raster_pre_snap

Name

    NV_conservative_raster_pre_snap

Name Strings

    GL_NV_conservative_raster_pre_snap

Contact

    Kedarnath Thangudu, NVIDIA Corporation (kthangudu 'at' nvidia.com)

Contributors

    Eric Werness, NVIDIA Corporation

Status

    Shipping in NVIDIA release 388.XX drivers and up

Version

    Last Modified Date:         November 15, 2017
    Revision:                   1

Number

    OpenGL Extension #517
    OpenGL ES Extension #297

Dependencies

    This extension is written against the NV_conservative_raster_pre_snap-
    _triangles extension as applied to OpenGL 4.3 specification 
    (Compatibility Profile) but may be used with the Core profile or OpenGL ES 
    2.0 or later.
    
Overview

    NV_conservative_raster_pre_snap_triangles provides a new mode to achieve
    rasterization of triangles that is conservative w.r.t the triangle at 
    infinite precision i.e. before it is snapped to the sub-pixel grid.  This
    extension provides a new mode that expands this functionality to lines and 
    points.

New Procedures and Functions

    None.
    
New Tokens

    Accepted by the <param> parameter of ConservativeRasterParameteriNV:
        CONSERVATIVE_RASTER_MODE_PRE_SNAP_NV            0x9550
    
Additions to Chapter 14 of the OpenGL 4.3 (Compatibility Profile) Specification
(Fixed-Function Primitive Assembly and Rasterization)

    Modify the paragraph describing ConservativeRasterParameteriNV in the 
    subsection 14.6.X "Conservative Rasterization" added by NV_conservative_-
    raster_pre_snap_triangles

    ... The <param> parameter specifies the conservative raster mode to be 
    used. If the mode is set to CONSERVATIVE_RASTER_MODE_POST_SNAP_NV, the 
    generated fragments are conservative w.r.t the primitive after it is 
    snapped to sub-pixel grid.  If the mode is set to CONSERVATIVE_RASTER_MODE_-
    PRE_SNAP_NV the fragments generated for a primitive will be conservative 
    w.r.t the primitive at infinite precision. Since non-degenerate 
    primitives may become degenerate due to vertex snapping, this mode will 
    generate fragments for zero length lines and zero area triangles which are 
    otherwise culled in the CONSERVATIVE_RASTER_MODE_POST_SNAP_NV. This mode 
    may also generate fragments for pixels that are within half a sub-pixel 
    distance away from the primitive at infinite precision.  If the mode is 
    set to CONSERVATIVE_RASTER_MODE_PRE_SNAP_TRIANGLES_NV, the pre-snap 
    conservative raster behavior described would apply only to triangles.  The 
    default mode is set to CONSERVATIVE_RASTER_MODE_POST_SNAP_NV.

    Modify the paragraphs describing conservative rasterization behavior for
    points, lines and polygons added by NV_conservative_raster as follows:

    If CONSERVATIVE_RASTERIZATION_NV is enabled, points are rasterized 
    according to point rasterization rules (section 14.4), except that a 
    fragment will be generated for a framebuffer pixel if the point's region 
    (a circle when MULTISAMPLING is enabled and POINT_SPRITE is disabled, or a 
    square otherwise) covers any portion of the pixel, including its edges or 
    points.  While conservative raster mode PRE_SNAP_NV respects the 
    MULTISAMPLE state, modes POST_SNAP_NV and PRE_SNAP_TRIANGLES_NV always use 
    point multisample rasterization rules (section 14.4.3), whether or not 
    MULTISAMPLE is actually enabled.  When performing conservative 
    rasterization of points, the POINT_SMOOTH enable is ignored and treated as
    disabled.

    If CONSERVATIVE_RASTERIZATION_NV is enabled, lines are rasterized 
    according to line rasterization rules (section 14.5), except that the 
    LINE_STIPPLE and LINE_SMOOTH enables are ignored and treated as disabled.  
    When the conservative raster mode is POST_SNAP or PRE_SNAP_TRIANGLES, 
    lines with zero length generate no fragments, while a fragment for the 
    pixel that contains the end points will be generated when the mode is 
    PRE_SNAP_NV.  Also, conservative raster mode PRE_SNAP_NV respects the 
    MULTISAMPLE state, while modes POST_SNAP_NV and PRE_SNAP_TRIANGLES_NV 
    always use line multisample rasterization rules (section 14.5.4), whether 
    or not MULTISAMPLE is actually enabled.

    If CONSERVATIVE_RASTERIZATION_NV is enabled, polygons are rasterized 
    according to polygon rasterization rules (section 14.6), except that 
    the POLYGON_SMOOTH enable is ignored and treated as disabled. 
    When the conservative raster mode is POST_SNAP_NV, polygons with 
    an area of zero generate no fragments, even for pixels that contain a 
    vertex or edge of the zero-area polygon, while modes PRE_SNAP_TRIANGLES 
    and PRE_SNAP_NV generate them.  Also, conservative raster mode PRE_SNAP_NV 
    respects the MULTISAMPLE state, while modes POST_SNAP_NV and 
    PRE_SNAP_TRIANGLES_NV always use polygon multisample rasterization rules 
    (section 14.6.6), whether or not MULTISAMPLE is actually enabled.

New State

    None.

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.
    
Modifications to the OpenGL Shading Language Specification, Version 4.30

    None.

Errors

    INVALID_ENUM is generated by ConservativeRasterParameteriNV if <pname> is
    not CONSERVATIVE_RASTER_MODE_NV, or if <param> is not CONSERVATIVE_RASTER_-
    MODE_POST_SNAP_NV, CONSERVATIVE_RASTER_MODE_PRE_SNAP_NV or CONSERVATIVE_-
    RASTER_MODE_PRE_SNAP_TRIANGLES_NV.

Issues

    (1) Would MODE_PRE_SNAP_NV generate fragments for zero width lines and
    zero diameter points?

    RESOLVED. No. Vertex snapping to the sub-pixel grid may cause in a line 
    to become zero length, so, MODE_PRE_SNAP will generate fragments for 
    zero length lines. Zero width lines and zero diameter points are culled as
    normal in both MODE_PRE_SNAP or MODE_POST_SNAP modes.

Revision History

    Revision 1
      - Internal revisions.

