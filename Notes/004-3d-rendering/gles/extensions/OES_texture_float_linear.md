# OES_texture_half_float_linear

Name

    OES_texture_half_float_linear
    OES_texture_float_linear

Name Strings

    GL_OES_texture_half_float_linear, GL_OES_texture_float_linear

Contact


Notice

    Copyright (c) 2005-2014 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

IP Status

    Please refer to the ARB_texture_float extension.

Status

    Ratified by the Khronos BOP, July 22, 2005.

Version

    Revision 3, September 21, 2018

Number

    OpenGL ES Extension #35

Dependencies

    This extension is written against the OpenGL ES 2.0 Specification.

    This extension is derived from the ARB_texture_float extension.

    Either
     - OES_texture_half_float and OES_texture_float are required.
    or
     - OpenGL ES 3.0 is required.

Overview

    These extensions expand upon the OES_texture_half_float and
    OES_texture_float extensions by allowing support for LINEAR
    magnification filter and LINEAR, NEAREST_MIPMAP_LINEAR,
    LINEAR_MIPMAP_NEAREST and LINEAR_MIPMAP_NEAREST minification
    filters.

    When implemented against OpenGL ES 3.0 or later versions, the
    existing sized 32-bit floating-point formats become texture-filterable,
    but no new formats are added.


(  Only when implemented against OpenGL ES 3.0  )

Additions to Chapter 3.8.3 of the OpenGL ES 3.0 Specification
(Texture Image Specification)

    Modify Table 3.13, Correspondence of sized internal color formats to base internal format:

    Check the ``Texture-filterable'' column for the R32F, RG32F, RGB32F, and RGBA32F formats.


Issues

   (1) Can you explain the interactions with OpenGL ES 3.x in more detail?

      OES_texture_float was written against OpenGL ES 2.0 and adds a set of
      new unsized formats, including floating-point versions of LUMINANCE and
      LUMINANCE_ALPHA formats.

      OES_texture_float_linear and OES_texture_half_float_linear makes these
      formats filterable for FLOAT and HALF_FLOAT_OES types, respectively.

      OpenGL ES 3.0 added sized internal formats. The unsized formats and
      the LUMINANCE formats are considered legacy formats. Floating point
      versions of these formats were therefore not added in OpenGL ES 3.0.

      Further, OpenGL ES 3.0 requires that the required floating-point
      formats with 16-bits per component ('half-float') are filterable, but
      does not support filtering for floating-point formats with 32-bits per
      component.

      Some OpenGL ES 3.0 implementations want a way to indicate that the
      required floating-point formats with 32-bit per component are also
      filterable _without_ adding the additional unsized formats from
      OES_texture_float. This is achieved by exposing this extension without
      exposing OES_texture_float.

      For an OpenGL ES 3.0 implementation, the following holds for the
      combination of the OES_texture_float, OES_texture_float_linear,
      and OES_texture_half_float_linear extensions:

        - If none of these extensions are supported:
           - floating-point formats with 32-bit per component are not filterable
           - floating-point formats with 16-bit per component are filterable

        - If OES_texture_float_linear is supported:
           - all floating-point formats in Table 3.13 are filterable

        - If OES_texture_float and OES_texture_float_linear are supported:
           - all floating-point formats in Table 3.13 are filterable
           - all formats of type FLOAT added by OES_texture_float are filterable

        - If OES_texture_half_float and OES_texture_half_float_linear are
           supported:
           - all floating-point formats in Table 3.13 are filterable
           - all formats of type HALF_FLOAT_OES added by OES_texture_half_float
             are filterable

New Procedures and Functions

   None

New Tokens

   None

Revision History

    07/06/2005  0.1   Original draft

    10/29/2014  2     Add interactions with ES 3.0/3.1.

    09/21/2018  3     Clarified interactions with ES 3.x.
