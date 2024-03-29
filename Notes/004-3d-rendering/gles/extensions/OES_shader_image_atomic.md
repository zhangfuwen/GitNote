# OES_shader_image_atomic

Name

    OES_shader_image_atomic

Name Strings

    GL_OES_shader_image_atomic

Contact

    Bill Licea-Kane, Qualcomm Technologies, Inc. ( billl 'at' qti.qualcomm.com )

Contributors

    Jeff Bolz, NVIDIA
    Pat Brown, NVIDIA
    Daniel Koch, NVIDIA
    Jon Leech
    Barthold Lichtenbelt, NVIDIA
    Bill Licea-Kane, AMD
    Eric Werness, NVIDIA
    Graham Sellers, AMD
    Greg Roth, NVIDIA
    Nick Haemel, AMD
    Pierre Boudier, AMD
    Piers Daniell, NVIDIA

Notice

    Copyright (c) 2011-2015 The Khronos Group Inc. Copyright termsat
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

    Complete.
    Ratified by the Khronos Board of Promoters on 2014/03/14.

Version

    Last Modified Date: April 30, 2015
    Revision: 5

Number

    OpenGL ES Extension #171

Dependencies

    This extension is written against the OpenGL ES Version 3.1 (April 29,
    2015) Specification.

    This extension is written against the OpenGL ES Shading Language 3.10,
    Revision 3, 6 June 2014 Specification.

    OpenGL ES 3.1 and GLSL ES 3.10 are required.

Overview

    This extension provides built-in functions allowing shaders to perform
    atomic read-modify-write operations to a single level of a texture
    object from any shader stage. These built-in functions are named
    imageAtomic*(), and accept integer texel coordinates to identify the
    texel accessed. These built-in functions extend the Images in ESSL 3.10.

Additions to Chapter 7 of the OpenGL ES 3.1 specification (Programs and Shaders)

    Change the first paragraph of section 7.10 "Images" on p. 113:

   "Images are special uniforms used in the OpenGL ES Shading Language to
    identify a level of a texture to be read or written using image load,
    store, and atomic built-in functions in the manner ..."


    Change the third paragraph of section 7.10:

   "The type of an image variable must match the texture target of the image
    currently bound to the image unit, otherwise the result of a load,
    store, or atomic operation..."


    Change the first sentence of section 7.11 "Shader Memory Access" on p.
    113:

   "... shader buffer variables, or to texture or buffer object memory using
    built-in image load, store, and atomic functions operating on shader
    image variables..."

Additions to Chapter 8 of the OpenGL ES 3.1 specification (Textures and Samplers)

    Change the caption to table 8.26, p. 199

   "Table 8.26: Mapping of image load, store, and atomic texel..."


    Change the first sentence of the second paragraph on p. 199:

   "When a shader accesses the texture bound to an image unit using a
    built-in image load, store or atomic function..."


    Change the fourth paragraph:

   "If the individual texel identified for an image load, store, or atomic
    operation ... Invalid image stores will have no effect. Invalid image
    atomics will not update any texture bound to the image unit and will
    return zero. An access is considered invalid if..."


    Change the first complete paragraph on p. 200:

   "Additionally, there are a number of cases where image load, store, or
    atomic operations are considered to involve a format mismatch. In such
    cases, undefined values will be returned by image loads and atomic
    operations, and undefined values will be written by stores and atomic
    operations. A format mismatch will occur if:"


    Change the last paragraph on p. 200:

   "Any image variable used for shader loads or atomic memory operations
    must be declared with a format <layout> qualifier matching ..."


    Change the last paragraph on p. 200:

   "When the format associated with an image unit does not exactly match the
    internal format of the texture bound to the image unit, image loads,
    stores and atomic operations re-interpret the memory holding the
    components of an accessed texel according to the format of the image
    unit. The re-interpretation for image loads and the read portion of
    image atomics is performed as though data were copied from the texel of
    the bound texture to a similar texel represented in the format of the
    image unit. Similarly, the re-interpretation for image stores and the
    write portion of image atomics is performed as though ..."

Changes to Chapter 20 (State Tables)

    Modify the description of MAX_IMAGE_UNITS in table 20.47 on p. 404:

   "MAX_IMAGE_UNITS ... No. of units for image load/store/atomics"

Additions to Appendix A of the OpenGL ES 3.1 specification (Invariance)

    Change the third sentence of the first paragraph in section A.1, p. 408:

   "This repeatability requirement doesn't apply when using shaders
    containing side effects (image stores, image atomic operations, atomic
    counter operations ..."


    Change the last sentence of Rule 4 in section A.3, p. 310:

   "Invariance is relaxed for shaders with side effects, such as image
    stores, image atomic operations, or accessing atomic counters (see
    section A.4)."


    Change the first sentence of the second paragraph of Rule 5, p. 310:

   "If a sequence of GL commands specifies primitives to be rendered with
    shaders containing side effects (image stores, image atomic operations,
    atomic counter operations ..."


    Change the first sentence of Rule 6, p. 411:

   "For any given GL and framebuffer state vector, and for any given GL
    command, the contents of any framebuffer state not directly or
    indirectly affected by results of shader image stores, image atomic
    operations, or atomic counter operations ..."


    Change the first bullet of Rule 7, p. 411:

   "* shader invocations do not use image atomic operations or atomic
      counters;"


    Change the first sentence of the second paragraph of Rule 7:

   "When any sequence of GL commands triggers shader invocations that
    perform image stores, image atomic operations, atomic counter
    operations ..."


Modifications to the OpenGL ES 3.10 Shading Language Specification

    Including the following line in a shader can be used to control the
    language features described in this extension:

        #extension GL_OES_shader_image_atomic : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

        #define GL_OES_shader_image_atomic


    Modifications to Chapter 4 (Variables and Types) of the OpenGL ES 3.10
    Shading Language Specification

    Change the first sentence of the third paragraph of section 4.1.7.2
    (Images), p. 28:

   "Image variables are used in the image load, store, and
    atomic functions described in section 8.12 ..."


    Add to section 8.12 (Image Functions)

    Add new overview and syntax and description table following existing
    Image Functions table, p. 129:

   "The atomic functions perform operations on individual texels or samples
    of an image variable. Atomic memory operations read a value from the
    selected texel, compute a new value using one of the operations
    described below, write the new value to the selected texel, and return
    the original value read. The contents of the texel being updated by the
    atomic operation are guaranteed not to be updated by any other image
    store or atomic function between the time the original value is read and
    the time the new value is written.

    Atomic memory operations are supported on only a subset of all image
    variable types; <image> must be either:

      * a signed integer image variable (type starts "iimage") and a format
        qualifier of "r32i", used with a <data> argument of type "int", or
      * an unsigned integer image variable (type starts "uimage") and a
        format qualifier of "r32ui", used with a <data> argument of type
        "uint", or
      * a float image variable (type starts "image") and a format qualifier
        of "r32f", used with a <data> argument of type "float"
        (imageAtomicExchange only).

    Add to the table of image functions, p. 129:

    Syntax                            Description
    -------------------------------   -------------------------------------
    highp uint imageAtomicAdd(        Computes a new value by adding the
        coherent IMAGE_PARAMS,        value of <data> to the contents of
        uint data);                   the selected texel.
    highp int imageAtomicAdd(
        coherent IMAGE_PARAMS,
        int data);
    -----------------------------------------------------------------------
    highp uint imageAtomicMin(        Computes a new value by taking the
        coherent IMAGE_PARAMS,        minimum of the value of <data> and
        uint data);                   the contents of the selected texel.
    highp int imageAtomicMin(
        coherent IMAGE_PARAMS,
        int data);
    -----------------------------------------------------------------------
    highp uint imageAtomicMax(        Computes a new value by taking the
        coherent IMAGE_PARAMS,        maximum of the value of <data> and
        uint data);                   the contents of the selected texel.
    highp int imageAtomicMax(
        coherent IMAGE_PARAMS,
        int data);
    -----------------------------------------------------------------------
    highp uint imageAtomicAnd(        Computes a new value by performing a
        coherent IMAGE_PARAMS,        bitwise AND of the value of <data>
        uint data);                   and the contents of the selected
    highp int imageAtomicAnd(         texel.
        coherent IMAGE_PARAMS,
        int data);
    -----------------------------------------------------------------------
    highp uint imageAtomicOr(         Computes a new value by performing a
        coherent IMAGE_PARAMS,        bitwise OR of the value of <data>
        uint data);                   and the contents of the selected
    highp int imageAtomicOr(          texel.
        coherent IMAGE_PARAMS,
        int data);
    -----------------------------------------------------------------------
    highp uint imageAtomicXor(        Computes a new value by performing a
        coherent IMAGE_PARAMS,        bitwise EXCLUSIVE OR of the value of
        uint data);                   <data> and the contents of the
    highp int imageAtomicXor(         selected texel.
        coherent IMAGE_PARAMS,
        int data);
    -----------------------------------------------------------------------
    highp uint imageAtomicExchange(   Computes a new value by simply
        coherent IMAGE_PARAMS,        copying the value of <data>.
        uint data);
    highp int imageAtomicExchange(
        coherent IMAGE_PARAMS,
        int data);
    highp float imageAtomicExchange(
        coherent IMAGE_PARAMS,
        float data);
    -----------------------------------------------------------------------
    highp uint imageAtomicCompSwap(   Compares the value of <compare> and
        coherent IMAGE_PARAMS,        the contents of the selected texel.
        uint compare,                 If the values are equal, the new
        uint data);                   value is given by <data>; otherwise,
    highp int imageAtomicCompSwap(    it is taken from the original value
        coherent IMAGE_PARAMS,        loaded from the texel.
        int compare,
        int data);
    -----------------------------------------------------------------------

Issues

    (0)  This extension is based on ARB_shader_image_load_store.  What
         are the major differences?

         1 - This extension splits out only the image atomic operations
             from ARB_shader_image_load_store.
         2 - It depends on image load/store functionality that is part of
             OpenGL ES 3.1, and inherits the list of differences from
             desktop implementations in ES 3.1. Note that issue 0 of the
             (unpublished) XXX_shader_image_load_store used for prototyping
             ES 3.1 language contains the list explicitly; perhaps it should
             should be reproduced here as well.

    (1) Should the shading language built-ins have OES suffixes?

        RESOLVED: No. Per Bug 11637, the WG made a policy decision
        that GLSL ES identifiers imported without semantic change
        or subsetting as OES extensions from core GLSL do not carry
        suffixes. The #extension mechanism must still be used to
        enable the appropriate extension before the functionality can
        be used.

Revision History

    Rev.  Date        Author    Changes
    1     2014-01-30  wwlk      Initial draft
    2     2014-01-31  wwlk      Update to current draft base specs
                                (further updates coming as
                                additional draft specs done.
                                Otherwise, this spec is probably
                                ready to review.)
    3     2014-02-11  dkoch     remove GLSL builtin suffixes per issue 1.
    4     2014-03-04  dkoch     add coherent to image parameter (Bug 11595)
    5     2015-04-30  Jon Leech Require OpenGL ES 3.1 and remove dependency
                                on nonexistent shader_image_load_store
                                extension. Rewrite changes against the
                                published ES 3.1 / ESSL 3.10 specifications.
                                Remove references to multisampled image
                                atomic support, which does not exist in ES
                                3.1. (Bugs 13879, 13887)
