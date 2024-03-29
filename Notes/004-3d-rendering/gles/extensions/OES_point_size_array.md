# OES_point_size_array

Name

    OES_point_size_array

Name Strings

    GL_OES_point_size_array

Contact

    Aaftab Munshi (amunshi@ati.com)

Notice

    Copyright (c) 2008-2013 The Khronos Group Inc. Copyright terms at
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

Version

    Last Modifed Date: 23 Dec 2008

Number

    OpenGL ES Extension #14

Dependencies

    OpenGL ES 1.0 is required.
    OES_point_sprite is required

    The extension is written against the OpenGL 1.5 Specification.

Overview

    This extension extends how points and point sprites are rendered
    by allowing an array of point sizes instead of a fixed input point
    size given by PointSize.  This provides flexibility for applications
    to do particle effects.

    The vertex arrays will be extended to include a point size array.
    The point size array can be enabled/disabled via POINT_SIZE_ARRAY_OES.

    The point size array, if enabled, controls the sizes used to render
    points and point sprites.  If point size array is enabled, the point
    size defined by PointSize is ignored.  The point sizes supplied in the 
    point size arrays will be the sizes used to render both points and
    point sprites.

IP Status

    None.

Issues

 
New Procedures and Functions

    void PointSizePointerOES(enum type, sizei stride, const void *ptr )

      valid values of type are GL_FIXED and GL_FLOAT
      the <size> parameter is removed since <size> is always 1

New Tokens

    Accepted by the <cap> parameters of EnableClientState/DisableClientState
    and by the <pname> parameter of IsEnabled:

      POINT_SIZE_ARRAY_OES          0x8B9C

    Accepted by the <pname> parameter of GetIntegerv:

      POINT_SIZE_ARRAY_TYPE_OES              0x898A
      POINT_SIZE_ARRAY_STRIDE_OES            0x898B
      POINT_SIZE_ARRAY_BUFFER_BINDING_OES    0x8B9F

    Accepted by the <pname> parameter of GetPointerv:

      POINT_SIZE_ARRAY_POINTER_OES  0x898C

Additions to Chapter 2 of the OpenGL 1.5 specification

    - section 2.8, added the following

            void PointSizePointerOES(enum type, sizei stride, const void *ptr);

            PointSizePointerOES is used to describe the point size for a given vertex

    - Added to table 2.4

                  Command                 Sizes       Types
                  -------                 -----       -----
                  PointSizePointerOES       1         float, fixed

    - (section 2.8), added the following
            Extend the cap flags passed to EnableClientState/DisableClientState
            to include POINT_SIZE_ARRAY_OES

Errors

    None.

New State

(table 6.6, p. 232)
                                   Get         Initial
Get Value                    Type  Command      Value   Description
---------                    ----  -------     -------  -----------
POINT_SIZE_ARRAY_OES          B    IsEnabled    False   point sprite array enable
POINT_SIZE_ARRAY_TYPE_OES     Z2   GetIntegerv  Float   type of point size
POINT_SIZE_ARRAY_STRIDE_OES   Z+   GetIntegerv  0       stride between point sizes
POINT_SIZE_ARRAY_POINTER_OES  Y    GetPointerv  0       pointer to point sprite array

(table 6.7, p. 233)

                                           Get         Initial
Get Value                            Type  Command     Value    Description
---------                            ----  -------     -------  -----------

POINT_SIZE_ARRAY_BUFFER_BINDING_OES  Z+    GetIntegerv    0     point size array
                                                                buffer binding

Revision History

    2008.12.23    benj            Per Bugzilla 4310, remove language saying that points
                                  are not drawn if point size array points are invalid,
                                  since this is not (always) detectable.  This aligns
                                  point size array behavior with all other array types,
                                  where implementations can crash if given invalid
                                  pointers to client data.
