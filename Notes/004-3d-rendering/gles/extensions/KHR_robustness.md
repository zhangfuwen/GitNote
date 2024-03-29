# KHR_robustness

Name

    KHR_robustness

Name Strings

    GL_KHR_robustness

Contributors

    Daniel Koch, NVIDIA
    Nicolas Capens, TransGaming
    Contributors to ARB_robustness

Contact

    Jon Leech (oddhack 'at' sonic.net) 
    Greg Roth, NVIDIA (groth 'at' nvidia.com)

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

    Version 11, August 21, 2014

Number

    ARB Extension #170
    OpenGL ES Extension #190

Dependencies

    OpenGL ES 2.0 or OpenGL 3.2 are required. Some features of this
    extension are supported by OpenGL ES only if OpenGL ES 3.0 or later is
    supported.

    EGL_EXT_create_context_robustness is used to determine if a context
    implementing this extension supports robust buffer access, and if it
    supports reset notification.

    This extension is written against the OpenGL ES 3.1 Specification
    (version of June 4, 2014) and the OpenGL ES 3.10.3 Shading Language
    Specification (version of June 6, 2014).

Overview

    Several recent trends in how OpenGL ES integrates into modern computer
    systems have created new requirements for robustness and security for GL
    rendering contexts.
    
    Additionally GPU architectures now support hardware fault detection;
    for example, video memory supporting ECC (error correcting codes)
    and error detection.  GL contexts should be capable of recovering
    from hardware faults such as uncorrectable memory errors.  Along with
    recovery from such hardware faults, the recovery mechanism can
    also allow recovery from video memory access exceptions and system
    software failures.  System software failures can be due to device
    changes or driver failures.

    GL queries that return (write) some number of bytes to a
    buffer indicated by a pointer parameter introduce risk of buffer
    overflows that might be exploitable by malware. To address this,
    queries with return value sizes that are not expressed directly by
    the parameters to the query itself are given additional API
    functions with an additional parameter that specifies the number of
    bytes in the buffer and never writing bytes beyond that limit. This
    is particularly useful for multi-threaded usage of GL contexts
    in a "share group" where one context can change objects in ways that
    can cause buffer overflows for another context's GL queries.

    The original ARB_vertex_buffer_object extension includes an issue
    that explicitly states program termination is allowed when
    out-of-bounds vertex buffer object fetches occur. Modern graphics
    hardware is capable of well-defined behavior in the case of out-of-
    bounds vertex buffer object fetches. Older hardware may require
    extra checks to enforce well-defined (and termination free)
    behavior, but this expense is warranted when processing potentially
    untrusted content.

    The intent of this extension is to address some specific robustness
    goals:

    *   For all existing GL queries, provide additional "safe" APIs 
        that limit data written to user pointers to a buffer size in 
        bytes that is an explicit additional parameter of the query.

    *   Provide a mechanism for a GL application to learn about
        graphics resets that affect the context.  When a graphics reset
        occurs, the GL context becomes unusable and the application
        must create a new context to continue operation. Detecting a
        graphics reset happens through an inexpensive query.

    *   Define behavior of OpenGL calls made after a graphics reset.

    *   Provide an enable to guarantee that out-of-bounds buffer object
        accesses by the GPU will have deterministic behavior and preclude
        application instability or termination due to an incorrect buffer
        access.  Such accesses include vertex buffer fetches of
        attributes and indices, and indexed reads of uniforms or
        parameters from buffers.

New Procedures and Functions

    NOTE: when implemented in an OpenGL ES context, all entry points defined
    by this extension must have a "KHR" suffix. When implemented in an
    OpenGL context, all entry points must have NO suffix, as shown below.

    enum GetGraphicsResetStatus();

    void ReadnPixels(int x, int y, sizei width, sizei height,
                        enum format, enum type, sizei bufSize,
                        void *data);

    void GetnUniformfv(uint program, int location, sizei bufSize,
                          float *params);
    void GetnUniformiv(uint program, int location, sizei bufSize,
                          int *params);
    void GetnUniformuiv(uint program, int location, sizei bufSize,
                           uint *params);


New Tokens

    NOTE: when implemented in an OpenGL ES context, all tokens defined by
    this extension must have a "_KHR" suffix. When implemented in an OpenGL
    context, all tokens must have NO suffix, as described below.

    Returned by GetGraphicsResetStatus:

        NO_ERROR                                        0x0000
        GUILTY_CONTEXT_RESET                            0x8253
        INNOCENT_CONTEXT_RESET                          0x8254
        UNKNOWN_CONTEXT_RESET                           0x8255

    Accepted by the <value> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv:

        CONTEXT_ROBUST_ACCESS                           0x90F3
        RESET_NOTIFICATION_STRATEGY                     0x8256

    Returned by GetIntegerv and related simple queries when <value> is
    RESET_NOTIFICATION_STRATEGY :

        LOSE_CONTEXT_ON_RESET                           0x8252
        NO_RESET_NOTIFICATION                           0x8261

    Returned by GetError:

        CONTEXT_LOST                                    0x0507


Additions to the OpenGL ES 3.1 Specification

    Add to section 2.3.1 "Errors" in the bullet list of implicit errors for
    GL commands on p. 14, and modify the following paragraph:

     * If the GL context has been reset as a result of a previous GL
       command, or if the context is reset as a side effect of execution of
       a command, a CONTEXT_LOST error is generated.

    The Specification attempts to explicitly describe these implicit error
    conditions (with the exception of OUT_OF_MEMORY [fn2] and CONTEXT_LOST
    [fn3]) wherever they apply. However ...

    [fn3] CONTEXT_LOST is not described because it occurs for reasons not
    directly related to the affected commands, and applies to almost all GL
    commands.


    Add to table 2.3 "Summary of GL errors" on p. 15:
    
    Error          Description               Offending command ignored?
    ------------   -----------------------   --------------------------
    CONTEXT_LOST   Context has been lost     Except as noted for
                   and reset by the driver   specific commands
    
    
    Add a new subsection 2.3.1rob after 2.3.1 "GL Errors", and renumber
    subsequent sections accordingly:

    2.3.1rob "Graphics Reset Recovery"

    Certain events can result in a reset of the GL context. After such an
    event, it is referred to as a <lost context> and is unusable for almost
    all purposes. Recovery requires creating a new context and recreating
    all relevant state from the lost context. The current status of the
    graphics reset state is returned by

        enum GetGraphicsResetStatus();

    The value returned indicates if the GL context has been in a reset state
    at any point since the last call to GetGraphicsResetStatus:

      * NO_ERROR indicates that the GL context has not been in a reset state
        since the last call.
      * GUILTY_CONTEXT_RESET indicates that a reset has been detected
        that is attributable to the current GL context.
      * INNOCENT_CONTEXT_RESET indicates a reset has been detected that
        is not attributable to the current GL context.
      * UNKNOWN_CONTEXT_RESET indicates a detected graphics reset whose
        cause is unknown.

    If a reset status other than NO_ERROR is returned and subsequent calls
    return NO_ERROR, the context reset was encountered and completed. If a
    reset status is repeatedly returned, the context may be in the process
    of resetting.

    Reset notification behavior is determined at context creation time, and
    may be queried by calling GetIntegerv with the symbolic constant
    RESET_NOTIFICATION_STRATEGY.

    If the reset notification behavior is NO_RESET_NOTIFICATION, then
    the implementation will never deliver notification of reset events, and
    GetGraphicsResetStatus will always return NO_ERROR[fn1].
       [fn1: In this case it is recommended that implementations should not
        allow loss of context state no matter what events occur. However,
        this is only a recommendation, and cannot be relied upon by
        applications.]

    If the behavior is LOSE_CONTEXT_ON_RESET, a graphics reset will
    result in a lost context and require creating a new context as described
    above. In this case GetGraphicsResetStatus will return an appropriate
    value from those described above.

    If a graphics reset notification occurs in a context, a notification
    must also occur in all other contexts which share objects with that
    context[fn2].
       [fn2: The values returned by GetGraphicsResetStatus in the
        different contexts may differ.]

    After a graphics reset has occurred on a context, subsequent GL commands
    on that context (or any context which shares with that context) will
    generate a CONTEXT_LOST error. Such commands will not have side effects
    (in particular, they will not modify memory passed by pointer for query
    results), and may not block indefinitely or cause termination of the
    application. Exceptions to this behavior include:

      * GetError and GetGraphicsResetStatus behave normally following a
        graphics reset, so that the application can determine a reset has
        occurred, and when it is safe to destroy and recreate the context.
      * Any commands which might cause a polling application to block
        indefinitely will generate a CONTEXT_LOST error, but will also
        return a value indicating completion to the application. Such
        commands include:

        + GetSynciv with <pname> SYNC_STATUS ignores the other parameters
          and returns SIGNALED in <values>.
        + GetQueryObjectuiv with <pname> QUERY_RESULT_AVAILABLE ignores the
          other parameters and returns TRUE in <params>.


    Modify section 7.12 "Shader, Program, and Program Pipeline Queries"
    on p. 125 to add the GetnUniform* commands:

    The commands

            [enumerate existing GetUniform*v commands, then add]

        void GetnUniformfv(uint program, int location, sizei bufSize,
                              float *params);
        void GetnUniformiv(uint program, int location, sizei bufSize,
                              int *params);
        void GetnUniformuiv(uint program, int location, sizei bufSize,
                               uint *params);

    return the value or values of the uniform at location <location> of the
    default uniform block for program object <program> in the array
    <params>. The type of the uniform at <location> determines the number of
    values returned. Calling GetnUniform*v ensures that no more than
    <bufSize> bytes are written into <params>.

    Add to the Errors section for Get*Uniform* on p. 126:

    An INVALID_OPERATION error is generated by GetnUniform* if the buffer
    size required to store the requested data is greater than <bufSize>


    Add subsection 10.3.1.1rob prior to section 10.3.2 "Vertex Attribute
    Divisors" on p. 241:

    10.3.1.1rob Robust Buffer Access Behavior

    Robust buffer access is enabled by creating a context with robust access
    enabled through the window system binding APIs. When enabled, indices
    within the element array that reference vertex data that lies outside
    the enabled attribute's vertex buffer object 
        [for OpenGL ES] result in undefined values
        [for OpenGL] result in reading zero
    for the corresponding attributes, but cannot result in application
    failure. 

    Robust buffer access behavior may be queried by calling
    GetIntegerv with the symbolic constant CONTEXT_ROBUST_ACCESS.


    Modify section 16.1.2 "ReadPixels" on p. 332 to add the ReadnPixels
    command:

    Pixels are read using

        void ReadPixels(int x, int y, sizei width, sizei height,
                        enum format, enum type, void *data);
        void ReadnPixels(int x, int y, sizei width, sizei height,
                            enum format, enum type, sizei bufSize,
                            void *data);

    The arguments after <x> and <y> to ReadPixels ... are summarized in
    table 16.1. ReadnPixels behaves identically to ReadPixels except that
    it does not write more than <bufSize> bytes into <data>.


    Add to the Errors section for ReadPixels and ReadnPixels:

    An INVALID_OPERATION error is generated by ReadnPixels if the buffer
    size required to store the requested data is greater than <bufSize>.


Additions to The OpenGL ES Shading Language Specification, Version 3.10

    Modify the first paragraph of section 4.1.9 "Arrays" on p. 30:

    Variables of the same type ... Undefined behavior results from indexing
    an array with a non-constant expression that is greater than or equal to
    the array size or less than 0. If robust buffer access is enabled (see
    section 10.3.1.1rob of the OpenGL ES 3.1 API Specification), such
    indexing must not result in abnormal program termination. The results
    are still undefined, but implementations are encouraged to produce zero
    values for such accesses. Arrays only have a single dimension ...

Dependencies on OpenGL ES

    For implementations against OpenGL ES, if OpenGL ES 3.0 or a later
    version is not supported, remove all references to GetnUniformuiv and
    remove the exceptional behavior of GetSynciv and GetQueryObjectuiv for
    lost contexts.

Interactions with EGL_EXT_create_context_robustness

    If the EGL window-system binding API is used to create a context, the
    EGL_EXT_create_context_robustness extension is supported, and the
    attribute EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT is set to EGL_TRUE when
    eglCreateContext is called, the resulting context will perform robust
    buffer access as described above in section 10.3.1.1rob, and the
    CONTEXT_ROBUST_ACCESS query will return GL_TRUE.

    If the EGL window-system binding API is used to create a context and the
    EGL_EXT_create_context_robustness extension is supported, then the value
    of attribute EGL_CONTEXT_RESET_NOTIFICATION_STRATEGY_EXT determines the
    reset notification behavior and the value of
    RESET_NOTIFICATION_STRATEGY, as described in section 2.3.1rob.

New Implementation Dependent State

    Get Value                   Type  Get Command Minimum Value     Description                 Sec.        Attribute
    --------------------------- ----  ----------- ----------------- --------------------------- ----------- ---------
    CONTEXT_ROBUST_ACCESS       B     GetIntegerv -                 Robust access enabled       10.3.1.1rob -
    RESET_NOTIFICATION_STRATEGY Z_2   GetIntegerv See sec. 2.3.1rob Reset notification behavior 2.3.1rob    -

Issues

    (Issues 2-8 are identical to those in the base EXT_robustness
    extension).

    1. What should this extension be called?

       RESOLVED: KHR_robustness

       This is just the OpenGL ES EXT_robustness extension (itself based on
       ARB_robustness) promoted to KHR status, with consistency with OpenGL
       4.5 behavior and phrasing included.

    2. How does this extension differ from Desktop GL's ARB_robustness?

       RESOLVED: Because EGL_EXT_create_context_robustness uses a separate
       attribute to enable robust buffer access, a corresponding query is
       added here.

       Also see issue 12.

    3. Should we provide a context creation mechanism to enable this
       extension?

       RESOLVED. Yes.

       Currently, EGL_EXT_create_context_robustness provides this mechanism
       via two unique attributes. These attributes differ from those
       specified by KHR_create_context to allow for differences in what
       functionality those attributes define.
        
    4. What can cause a graphics reset?

       Either user or implementor errors may result in a graphics reset. If
       the application attempts to perform a rendering that takes too long
       whether due to an infinite loop in a shader or even just a rendering
       operation that takes too long on the given hardware. Implementation
       errors may produce badly formed hardware commands. Memory access
       errors may result from user or implementor mistakes. On some systems,
       power management events such as system sleep, screen saver
       activation, or pre-emption may also context resets to occur. Any of
       these events may result in a graphics reset event that will be
       detectable by the mechanism described in this extension.

    5. How should the application react to a reset context event?

       RESOLVED: For this extension, the application is expected to query
       the reset status until NO_ERROR is returned. If a reset is
       encountered, at least one *RESET* status will be returned. Once
       NO_ERROR is again encountered, the application can safely destroy the
       old context and create a new one.

       After a reset event, apps should not use a context for any purpose
       other than determining its reset status (using either GetError to
       identify a CONTEXT_LOST error, or GetGraphicsResetStatus()), and
       then destroying it. If a context receives a reset event, all other
       contexts in its share group will also receive reset events, and
       should be destroyed and recreated.

       Apps should be cautious in interpreting the GUILTY and INNOCENT reset
       statuses. These are guidelines to the immediate cause of a reset, but
       not guarantees of the ultimate cause.

    6. If a graphics reset occurs in a shared context, what happens in
       shared contexts?

       RESOLVED: A reset in one context will result in a reset in all other
       contexts in its share group.

    7. How can an application query for robust buffer access support, since
       this is now determined at context creation time?

       RESOLVED. The application can query the value of
       CONTEXT_ROBUST_ACCESS using GetIntegerv. If true, this
       functionality is enabled.

    8. How is the reset notification behavior controlled?

       RESOLVED: Reset notification behavior is determined at context
       creation time using EGL/GLX/WGL/etc. mechanisms. In order that shared
       objects be handled predictably, a context cannot share with another
       context unless both have the same reset notification behavior.

    9. How does this extension differ from EXT_robustness?

       RESOLVED: By adding the new CONTEXT_LOST error, removing support for
       ES 1.1 (for logistical reasons), and changing suffixes from EXT to
       KHR (for ES) or no suffix (for GL).

   10. How should this extension be enabled via EGL?

       PROPOSED: Either by using EGL 1.5 context creation APIs (see section
       3.7.1.5 of the EGL 1.5 spec), or EGL_EXT_create_context_robustness.
       This interaction will be noted in both the EGL 1.5 spec and the
       extension spec by allowing at least one, and possibly both of
       GL_EXT_robustness and GL_KHR_robustness to be supported by contexts
       created via these mechanisms. If a context supporting
       GL_KHR_robustness is created, it will optionally support
       GL_KHR_robust_buffer_access_behavior as well (see issue 5 of that
       extension).

   11. What should the behavior of GL calls made after a context is lost be?
       This can be expected to occur when the context has been lost, but the
       app hasn't polled the reset status yet.

       RESOLVED: Set a new CONTEXT_LOST error on all GL calls (except
       GetError and GetGraphicsReset).

       DISCUSSION: GetError and GetGraphicsResetStatus must continue to
       work at least to the extent of their interactions with robustness
       features, so that apps can determine a context was lost and see the
       effect on other commands. Commands which might block indefinitely or
       cause the app to block indefinitely while polling are defined to
       return immediately, with values which should end a polling loop. Such
       commands include sync and query object queries for completion.

       Special handling of these commands is defined to return values which
       should not cause blocking, but also to generate the CONTEXT_LOST
       error. This is intended to deal well both with apps that are polling
       on a query value, and apps that may test for the error immediately
       after each command (such as debuggers). There has been no pushback
       against

       We believe there are no other commands requiring this special
       handling in the core API. ClientWaitSync and WaitSync were also
       proposed, but with the current language (which specifies that
       commands on a lost context may not block) should not need exceptional
       handling.

       REJECTED OPTIONS:

       A) GL will try to handle the call as best as possible. Depending on
       the call this may use GPU resources which may now have undefined
       contents.

       B) GL calls become no-ops. This would be like having a null-dispatch
       table installed when you don't have a current context. The problem
       with this approach is that any call that has return parameters won't
       fill them in and the application won't be able to detect that a value
       was not returned.

       REJECTED SUB-FEATURES:

       We discussed using OUT_OF_MEMORY or INVALID_OPERATION. Both are
       misleading and don't uniquely identify the problem as a runtime error
       out of scope of anything the app did.

       We discussed allowing all commands to have side effects in addition
       to generating CONTEXT_LOST, such as putting a "safe" value into
       return parameters. Without compelling reason to allow this behavior,
       it is cleaner to define the spec to leave results unchanged aside
       from the exceptional cases intended to prevent hanging on polling
       loops.

   12. What changed in promoting OES_robustness to KHR_robustness? What
       remains to be done for consistency between GL and ES?

       DISCUSSION: The ARB agreed to support robustness and
       robust_buffer_access_behavior as a KHR extension, with the following
       chnages and issues to be resolved during the ratification period
       (none are IP-related):

       a) As was done for KHR_debug, the extension is defined to have KHR
          suffixes only in ES implementations. GL implementations do not
          have suffixes. This is so KHR_robustness can be used as a
          backwards-compatibility extension for OpenGL 4.5.

       b) Minor spec language tweaks were done for consistency with OpenGL
          4.5, to eliminate redundancy. None are functional with the
          exception of a change in section 10.3.1.1rob, which imposes
          stronger requirements on out-of-bounds indexed attribute access
          for GL (returns zero) relative to ES (undefined return values).

       c) The ARB wants to add these commands to the extension, as defined
          in OpenGL 4.5. They would be only be supported in GL contexts:

            GetnUniformdv
            GetnTexImage
            GetnCompressedTexImage

          We do not think we need to add the compatibility-mode Getn*
          queries defined by ARB_robustness.

       d) OpenGL 4.5 exposes support for robustness by setting
          CONTEXT_FLAG_ROBUST_ACCESS_BIT in the CONTEXT_FLAGS query. ES (and
          this extension) expose it with the CONTEXT_ROBUST_ACCESS query.
          Jon proposes we resolve this by only defining the CONTEXT_FLAGS
          query for GL, but defining the CONTEXT_ROBUST_ACCESS query for
          both GL and ES, and aliasing the flag bit with the access boolean.
          This will result in minor changes to both GL 4.5 and this
          extension.

       e) This extension modifies the Shading Language specification in
          section 4.1.9 to restrict behavior of out of bounds array
          accesses. GLSL 4.50 has a considerably broader section 5.11
          "Out-of-Bounds Accesses" which restricts behavior of arrays,
          vectors, structs, etc. We will want to include that language in
          KHR_robustness at least as GLSL-specific language; if ES wants to
          adopt the broader language for GLSL-ES that might still be doable.

Revision History

    Rev.    Date       Author     Changes
    ----  ------------ ---------  ------------------------------------------
     11   2014/08/21   Jon Leech  Fix typo.
     10   2014/06/26   Jon Leech  Change from OES to KHR. Update issues 1
                                  and 9; add issue 12 on ES / GL
                                  differences.
      9   2014/06/26   Jon Leech  Minor phrasing fixes to be more consistent
                                  with equivalent GL functionality.
      8   2014/06/24   Jon Leech  Fix typos & mention GetError in issue 5.
                                  Reorder API spec edits in section order.
                                  Resolve issues 9-11 (Bug 12104 comments
                                  #42-45). Assign CONTEXT_LOST enum.
      7   2014/06/02   Jon Leech  Remove "safe" return value behavior for
                                  queries when contexts are lost (Bug 12104
                                  comment #27).
      6   2014/06/02   Jon Leech  Rebase extension on OpenGL ES 3.1 specs
                                  and reflow paragraphs (no functionality
                                  changes are made in this version).
      5   2014/05/16   Jon Leech  Add CONTEXT_LOST error and exceptional
                                  behavior for potentially-blocking
                                  commands (Bug 8411).
      4   2014/05/14   Jon Leech  Add issue 11 on behavior of GL calls made
                                  after a context is lost.
      3   2014/05/07   Jon Leech  Add GetnUniformuivOES for ES 3.0.
      2   2014/04/26   Jon Leech  Updates based on Ken's comments in bug
                                  12104 and to reference EGL 1.5.
      1   2014/04/23   Jon Leech  Branch from EXT_robustness version 3 and
                                  promote to OES suffixes. Add issues 9-10.
