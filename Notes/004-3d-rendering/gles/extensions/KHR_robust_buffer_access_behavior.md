# KHR_robust_buffer_access_behavior

Name

    KHR_robust_buffer_access_behavior

Name Strings

    GL_KHR_robust_buffer_access_behavior

Contact

    Jon Leech (oddhack 'at' sonic.net)
    Piers Daniell, NVIDIA (pdaniell 'at' nvidia.com)

Contributors

    Jan-Harald Fredriksen, ARM
    Jeff Bolz, NVIDIA
    Kenneth Russell, Google
    Pat Brown, NVIDIA

Notice

    Copyright (c) 2012-2014 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL and OpenGL ES Working Groups. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Complete. 
    Approved by the OpenGL ES Working Group on June 25, 2014.
    Approved by the ARB on June 26, 2014.
    Ratified by the Khronos Board of Promoters on August 7, 2014.

Version

    Version 7, June 26, 2014

Number

    ARB Extension #169
    OpenGL ES Extension #189

Dependencies

    OpenGL ES 2.0 or OpenGL 3.2 are required.

    GL_KHR_robustness is required.
    
    This extension is written against the OpenGL ES 3.1 Specification
    (version of June 4, 2014) and the OpenGL ES 3.10.3 Shading Language
    Specification (version of June 6, 2014)

Overview

    This extension specifies the behavior of out-of-bounds buffer and
    array accesses. This is an improvement over the existing
    KHR_robustness extension which states that the application should
    not crash, but that behavior is otherwise undefined. This extension
    specifies the access protection provided by the GL to ensure that
    out-of-bounds accesses cannot read from or write to data not owned
    by the application. All accesses are contained within the buffer
    object and program area they reference. These additional robustness
    guarantees apply to contexts created with the robust access flag
    set.

IP Status

    No known IP claims.

New Procedures and Functions

    None

New Tokens

    None

Additions to the OpenGL ES 3.1 Specification

    Append to section 6.4 "Effects of Accessing Outside Buffer Bounds" on p.
    58:
    
   "Robust buffer access can be enabled by creating a context with robust
    access enabled through the window system binding APIs. When enabled, any
    command unable to generate a GL error as described above, such as buffer
    object accesses from the active program, will not read or modify memory
    outside of the data store of the buffer object and will not result in GL
    interruption or termination.

    Out-of-bounds reads may return any of the following values:

      * Values from anywhere within the buffer object.
      * Zero values, or (0,0,0,x) vectors for vector reads where x is a
        valid value represented in the type of the vector components and may
        be any of

        + 0, 1, or the maximum representable positive integer value, for
          signed or unsigned integer components
        + 0.0 or 1.0, for floating-point components

    Out-of-bounds writes may modify values within the buffer object or be
    discarded.

    Accesses made through resources attached to binding points are only
    protected within the buffer object from which the binding point is
    declared. For example, for an out-of-bounds access to a member variable
    of a uniform block, the access protection is provided within the uniform
    buffer object, and not for the bound buffer range for this uniform
    block."


    Add a new subsection 10.3.4rob "Robust Buffer Access" preceding section
    10.3.5 "Packed Vertex Data Formats" on p. 243:

   "Robust buffer access can be enabled by creating a context with robust
    access enabled through the window system binding APIs. When enabled,
    indices within the element array (see section 10.3.7) that reference
    vertex data that lies outside the enabled attribute's vertex buffer
    object result in reading zero. It is not possible to read vertex data
    from outside the enabled vertex buffer objects or from another GL
    context, and these accesses do not result in abnormal program
    termination."


    Replace the last paragraph of section 11.1.3.2 "Texel Fetches", on p.
    265:
    
    "In all the above cases, if the context was created with robust buffer
    access enabled (see section 10.3.4rob), the result of the texture fetch
    is zero, or a texture source color of (0,0,0,1) in the case of a texel
    fetch from an incomplete texture. If robust buffer access is not
    enabled, the result of the texture fetch is undefined in each case."


    Replace the last paragraph of section 11.1.3.12 "Undefined Behavior" on
    p. 272:
    
   "Robust buffer access can be enabled by creating a context with
    robust access enabled through the window system binding APIs. When
    enabled, out-of-bounds accesses will be bounded within the working
    memory of the active program, cannot access memory owned by other
    GL contexts, and will not result in abnormal program termination.
    Out-of-bounds access to local and global variables cannot read
    values from other program invocations. An out-of-bounds read may
    return another value from the active program's working memory or
    zero. An out-of-bounds write may overwrite a value from the active
    program's working memory or be discarded.
    
    Out-of-bounds accesses to resources backed by buffer objects cannot read
    or modify data outside of the buffer object. For resources bound to
    buffer ranges, access is restricted within the buffer object from which
    the buffer range was created, and not within the buffer range itself.
    Out-of-bounds reads and writes behave as described in section 6.4.
    
    Out-of-bounds accesses to arrays of resources, such as an array of
    textures, can only access the data of bound resources. Reads from
    unbound resources return zero and writes are discarded. It is not
    possible to access data owned by other GL contexts.

    Applications that require defined behavior for out-of-bounds
    accesses should range check all computed indices before
    dereferencing the array, vector or matrix."


Additions to chapter 5 of the OpenGL ES Shading Language Specification
version 3.10.3

    Add a new section 5.12 "Out-of-Bounds Access and Robust Buffer Access
    Behavior" on p. 84:
    
   "In the sections described above for array, vector, matrix and structure
    accesses, any out-of-bounds access produces undefined behavior. However,
    if robust buffer access is enabled via the GL API, such accesses will be
    bound within the memory extent of the active program. It will not be
    possible to access memory from other programs, and accesses will not
    result in abnormal program termination. Out-of-bounds reads return
    undefined values, which include values from other variables of the
    active program or zero. Out-of-bounds writes may be discarded or
    overwrite other variables of the active program, depending on the value
    of the computed index and how this relates to the extent of the active
    program's memory. Applications that require defined behavior for
    out-of-bounds accesses should range check all computed indices before
    dereferencing an array."

Errors

    None

New State

    None

New Implementation Dependent State

    None

Interactions with OpenGL ES 2.0

    If only OpenGL ES 2.0 is supported then modifications to texel fetch
    behavior are ignored, since texel fetch functionality does not exist
    in OpenGL ES 2.0.

Interactions with OpenGL

    In OpenGL implementations of this extension, the language in section 6.4
    on values returned from out-of-bound reads is still applied in its
    entirety to the "Undefined Behavior" language in section 11.1.3.12,
    including the (0,0,0,x) read behavior.

    However, when read specifically against the operations in 6.4,
    out-of-bounds reads are further restricted to return either values from
    anywhere within the buffer object, or zero. The less tightly specified
    (0,0,0,x) reads defined for OpenGL ES does *not* apply in this case.

Issues

    1) Why are out-of-bounds reads of buffer object backed resources defined
       to allow returning "a value from the current buffer object" in
       addition to one of several possible defined values?
       
       RESOLVED: This is necessary to allow the implementation to either
       range check, apply a mask/modulus or apply a clamp to the index.
       
    2) Why do out-of-bounds writes to buffer object backed resources have a
       stronger guarantee than reads? The spec says that writes outside of
       the bounded range are discarded but reads are only protected within
       the buffer object.
       
       RESOLVED: This stronger guarantee for writes can be made because
       GPUs made since around 2008 (DX10) already implement this behavior.

    3) How does this extension differ from
       ARB_robust_buffer_access_behavior?

       - It is written against OpenGL ES 3.1 instead of GL 4.2, and can
         be implemented for OpenGL ES 2.0 contexts as well.
       - References to GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT_ARB are
         removed, since there is no query for context creation flags in
         any version of OpenGL ES.
       - For OpenGL ES implementations only, it widens the scope of possible
         return values from OOB buffer reads to (0,0,0,x) where x is zero,
         one, or MAXINT for integer types.

    4) What value should be returned for out of bounds reads that are
       not part of another resource?

       DISCUSSION: As noted in bug 12104, some implementations cannot return
       zero for all components in this case, but may return another defined
       value, such as one, for the alpha component of vector reads. The
       agreed resolution has been incorporated in the language for section
       2.9.4 and referenced from elsewhere in the extension language.

    5) How should this extension be enabled via EGL?

       PROPOSED: If a context is successfully created supporting
       KHR_robustness (see issue 10 of that spec), the EGL 1.5 spec and
       EGL_EXT_create_context_robustness extensions will be modified to
       *allow* (but not require) support of
       GL_KHR_robust_buffer_access_behavior as well.

       DISCUSSION: We can't require support of this extension, because
       that's a behavior modification to the EGL context creation
       functionality and cannot be supported on existing implementations
       which may support robustness already but not the additional
       guarantees of this spec.

    6) What changed in promoting this extension from OES to KHR? What
       remains to be done for consistency between GL and ES?

       DISCUSSION: The only meaningful difference is identified in the
       "Interactions with OpenGL" section, and is simply slightly tighter
       constrains on out-of-range buffer reads through the operations in
       section 6.4.


Revision History

    Rev.    Date       Author     Changes
    ----  ------------ ---------  -------------------------------------------
      7   2014/06/26   Jon Leech  Change from OES to KHR. Update issue 3
                                  and add issue 6 on ES / GL differences.
      6   2014/06/24   Jon Leech  Rebase against OpenGL ES 3.1. Fix typos.
                                  Add issue 5 on enabling the extension
                                  at context creation time.
      5   2014/06/09   Jon Leech  Update values-returned language to allow
                                  returning any of 0, 1, or the maximum 
                                  representable positive value for integer
                                  types (Bug 12104).
      4   2014/06/02   Jon Leech  Incorporate new language on values returned
                                  from reads inside a buffer object (Bug
                                  12104 comment #31).
      3   2014/05/14   Jon Leech  Revert language on disabled attribute 
                                  reads mistakenly included from bug 10695.
      2   2014/05/07   Jon Leech  Add issue 4 on values to be returned
                                  from out of bounds reads (Bug 12104).
      1   2014/04/23   Jon Leech  Branch from ARB_rbab and convert to be
                                  based on ES 3 specs instead of GL 4.2.
