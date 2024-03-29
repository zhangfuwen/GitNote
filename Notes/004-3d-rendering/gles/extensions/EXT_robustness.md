# EXT_robustness

Name

    EXT_robustness

Name Strings

    GL_EXT_robustness

Contributors

    Daniel Koch, TransGaming
    Nicolas Capens, TransGaming
    Contributors to ARB_robustness

Contact

    Greg Roth, NVIDIA (groth 'at' nvidia.com)

Status

    Complete.

Version

    Version 3, 2011/10/31

Number

    OpenGL ES Extension #107

Dependencies

    This extension is written against the OpenGL ES 2.0 Specification
    but can apply to OpenGL ES 1.1 and up.

    EGL_EXT_create_context_robustness is used to determine if a context
    implementing this extension supports robust buffer access, and if it
    supports reset notification. 

Overview

    Several recent trends in how OpenGL integrates into modern computer
    systems have created new requirements for robustness and security
    for OpenGL rendering contexts.
    
    Additionally GPU architectures now support hardware fault detection;
    for example, video memory supporting ECC (error correcting codes)
    and error detection.  OpenGL contexts should be capable of recovering
    from hardware faults such as uncorrectable memory errors.  Along with
    recovery from such hardware faults, the recovery mechanism can
    also allow recovery from video memory access exceptions and system
    software failures.  System software failures can be due to device
    changes or driver failures.

    OpenGL queries that that return (write) some number of bytes to a
    buffer indicated by a pointer parameter introduce risk of buffer
    overflows that might be exploitable by malware. To address this,
    queries with return value sizes that are not expressed directly by
    the parameters to the query itself are given additional API
    functions with an additional parameter that specifies the number of
    bytes in the buffer and never writing bytes beyond that limit. This
    is particularly useful for multi-threaded usage of OpenGL contexts
    in a "share group" where one context can change objects in ways that
    can cause buffer overflows for another context's OpenGL queries.

    The original ARB_vertex_buffer_object extension includes an issue
    that explicitly states program termination is allowed when
    out-of-bounds vertex buffer object fetches occur. Modern graphics
    hardware is capable well-defined behavior in the case of out-of-
    bounds vertex buffer object fetches. Older hardware may require
    extra checks to enforce well-defined (and termination free)
    behavior, but this expense is warranted when processing potentially
    untrusted content.

    The intent of this extension is to address some specific robustness
    goals:

    *   For all existing OpenGL queries, provide additional "safe" APIs 
        that limit data written to user pointers to a buffer size in 
        bytes that is an explicit additional parameter of the query.

    *   Provide a mechanism for an OpenGL application to learn about
        graphics resets that affect the context.  When a graphics reset
        occurs, the OpenGL context becomes unusable and the application
        must create a new context to continue operation. Detecting a
        graphics reset happens through an inexpensive query.

    *   Provide an enable to guarantee that out-of-bounds buffer object
        accesses by the GPU will have deterministic behavior and preclude
        application instability or termination due to an incorrect buffer
        access.  Such accesses include vertex buffer fetches of
        attributes and indices, and indexed reads of uniforms or
        parameters from buffers.

New Procedures and Functions

        enum GetGraphicsResetStatusEXT();

        void ReadnPixelsEXT(int x, int y, sizei width, sizei height,
                            enum format, enum type, sizei bufSize,
                            void *data);

        void GetnUniformfvEXT(uint program, int location, sizei bufSize,
                              float *params);
        void GetnUniformivEXT(uint program, int location, sizei bufSize,
                              int *params);

New Tokens

    Returned by GetGraphicsResetStatusEXT:

        NO_ERROR                                        0x0000
        GUILTY_CONTEXT_RESET_EXT                        0x8253
        INNOCENT_CONTEXT_RESET_EXT                      0x8254
        UNKNOWN_CONTEXT_RESET_EXT                       0x8255

    Accepted by the <value> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv:

        CONTEXT_ROBUST_ACCESS_EXT                       0x90F3
        RESET_NOTIFICATION_STRATEGY_EXT                 0x8256

    Returned by GetIntegerv and related simple queries when <value> is
    RESET_NOTIFICATION_STRATEGY_EXT :

        LOSE_CONTEXT_ON_RESET_EXT                       0x8252
        NO_RESET_NOTIFICATION_EXT                       0x8261

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL ES Operation)

Add a new subsection after 2.5 "GL Errors" and renumber subsequent
sections accordingly.

    2.6 "Graphics Reset Recovery"

    Certain events can result in a reset of the GL context. Such a reset
    causes all context state to be lost. Recovery from such events
    requires recreation of all objects in the affected context. The
    current status of the graphics reset state is returned by

        enum GetGraphicsResetStatusEXT();

    The symbolic constant returned indicates if the GL context has been
    in a reset state at any point since the last call to
    GetGraphicsResetStatusEXT. NO_ERROR indicates that the GL context
    has not been in a reset state since the last call.
    GUILTY_CONTEXT_RESET_EXT indicates that a reset has been detected
    that is attributable to the current GL context.
    INNOCENT_CONTEXT_RESET_EXT indicates a reset has been detected that
    is not attributable to the current GL context.
    UNKNOWN_CONTEXT_RESET_EXT indicates a detected graphics reset whose
    cause is unknown.

    If a reset status other than NO_ERROR is returned and subsequent
    calls return NO_ERROR, the context reset was encountered and
    completed. If a reset status is repeatedly returned, the context may
    be in the process of resetting.

    Reset notification behavior is determined at context creation time,
    and may be queried by calling GetIntegerv with the symbolic constant
    RESET_NOTIFICATION_STRATEGY_EXT.

    If the reset notification behavior is NO_RESET_NOTIFICATION_EXT,
    then the implementation will never deliver notification of reset
    events, and GetGraphicsResetStatusEXT will always return
    NO_ERROR[fn1].
       [fn1: In this case it is recommended that implementations should
        not allow loss of context state no matter what events occur.
        However, this is only a recommendation, and cannot be relied
        upon by applications.]

    If the behavior is LOSE_CONTEXT_ON_RESET_EXT, a graphics reset will
    result in the loss of all context state, requiring the recreation of
    all associated objects. In this case GetGraphicsResetStatusEXT may
    return any of the values described above.

    If a graphics reset notification occurs in a context, a notification
    must also occur in all other contexts which share objects with that
    context[fn2].
       [fn2: The values returned by GetGraphicsResetStatusEXT in the
        different contexts may differ.]

    Add to Section 2.8 "Vertex Arrays" before subsection "Transferring
    Array Elements"

    Robust buffer access is enabled by creating a context with robust
    access enabled through the window system binding APIs. When enabled,
    indices within the vertex array that lie outside the arrays defined
    for enabled attributes result in undefined values for the
    corresponding attributes, but cannot result in application failure.
    Robust buffer access behavior may be queried by calling GetIntegerv
    with the symbolic constant CONTEXT_ROBUST_ACCESS_EXT.

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Frame Buffer)

    Modify section 4.3.1 "Reading Pixels"

    Pixels are read using

        void ReadPixels(int x, int y, sizei width, sizei height,
                        enum format, enum type, void *data);
        void ReadnPixelsEXT(int x, int y, sizei width, sizei height,
                           enum format, enum type, sizei bufSize,
                           void *data);

    Add to the description of ReadPixels:

    ReadnPixelsEXT behaves identically to ReadPixels except that it does
    not write more than <bufSize> bytes into <data>. If the buffer size
    required to fill all the requested data is greater than <bufSize> an
    INVALID_OPERATION error is generated and <data> is not altered.

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special
Functions):

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and
State Requests)

    Modify Section 6.1.8 "Shader and Program Queries"

    The commands

        void GetUniformfv(uint program, int location, float *params);
        void GetnUniformfvEXT(uint program, int location, sizei bufSize,
                              float *params);
        void GetUniformiv(uint program, int location, int *params);
        void GetnUniformivEXT(uint program, int location, sizei bufSize,
                              int *params);

    return the value or values of the uniform at location <location>
    for program object <program> in the array <params>. Calling
    GetnUniformfvEXT or GetnUniformivEXT ensures that no more than
    <bufSize> bytes are written into <params>. If the buffer size
    required to fill all the requested data is greater than <bufSize> an
    INVALID_OPERATION error is generated and <params> is not altered.
    ...

Additions to The OpenGL ES Shading Language Specification, Version 1.

    Append to the third paragraph of section 4.1.9 "Arrays"

    If robust buffer access is enabled via the OpenGL ES API, such
    indexing must not result in abnormal program termination. The
    results are still undefined, but implementations are encouraged to
    produce zero values for such accesses.

Interactions with EGL_EXT_create_context_robustness

    If the EGL window-system binding API is used to create a context,
    the EGL_EXT_create_context_robustness extension is supported, and
    the attribute EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT is set to
    EGL_TRUE when eglCreateContext is called, the resulting context will
    perform robust buffer access as described above in section 2.8, and
    the CONTEXT_ROBUST_ACCESS_EXT query will return GL_TRUE as described
    above in section 6.1.5.

    If the EGL window-system binding API is used to create a context and
    the EGL_EXT_create_context_robustness extension is supported, then
    the value of attribute EGL_CONTEXT_RESET_NOTIFICATION_STRATEGY_EXT
    determines the reset notification behavior and the value of
    RESET_NOTIFICATION_STRATEGY_EXT, as described in section 2.6.

Errors

    ReadnPixelsEXT, GetnUniformfvEXT, and GetnUniformivEXT share all the
    errors of their unsized buffer query counterparts with the addition
    that INVALID_OPERATION is generated if the buffer size required to
    fill all the requested data is greater than <bufSize>.

New Implementation Dependent State

    Get Value                       Type  Get Command     Minimum Value    Description                  Sec.  Attribute
    ---------                       ----  -----------     -------------    ---------------------------  ----- ---------
    CONTEXT_ROBUST_ACCESS_EXT       B     GetIntegerv     -                Robust access enabled        6.1.5 -
    RESET_NOTIFICATION_STRATEGY_EXT Z_2   GetIntegerv     See sec. 2.6     Reset notification behavior  2.6   -

Issues


    1.  What should this extension be called?

        RESOLVED: EXT_robustness

        Since this is intended to be a version of ARB_robustness for
        OpenGL ES, it should be named accordingly.

    2.  How does this extension differ from Desktop GL's ARB_robustness?

        RESOLVED: Because EGL_EXT_create_context_robustness uses a
	separate attribute to enable robust buffer access, a
	corresponding query is added here.

    3.  Should we provide a context creation mechanism to enable this extension?

        RESOLVED. Yes.

        Currently, EGL_EXT_create_context_robustness provides this
        mechanism via two unique attributes. These attributes differ
	from those specified by KHR_create_context to allow for
	differences in what functionality those attributes define.
        
    4. What can cause a graphics reset?

       Either user or implementor errors may result in a graphics reset.
       If the application attempts to perform a rendering that takes too long
       whether due to an infinite loop in a shader or even just a rendering
       operation that takes too long on the given hardware. Implementation
       errors may produce badly formed hardware commands. Memory access errors
       may result from user or implementor mistakes. On some systems, power
       management events such as system sleep, screen saver activation, or
       pre-emption may also context resets to occur. Any of these events may
       result in a graphics reset event that will be detectable by the
       mechanism described in this extension.

    5. How should the application react to a reset context event?

       RESOLVED: For this extension, the application is expected to query
       the reset status until NO_ERROR is returned. If a reset is encountered,
       at least one *RESET* status will be returned. Once NO_ERROR is again
       encountered, the application can safely destroy the old context and
       create a new one.

       After a reset event, apps should not use a context for any purpose
       other than determining its reset status, and then destroying it. If a
       context receives a reset event, all other contexts in its share group
       will also receive reset events, and should be destroyed and
       recreated.

       Apps should be cautious in interpreting the GUILTY and INNOCENT reset
       statuses. These are guidelines to the immediate cause of a reset, but
       not guarantees of the ultimate cause.

    6. If a graphics reset occurs in a shared context, what happens in
       shared contexts?

       RESOLVED: A reset in one context will result in a reset in all other
       contexts in its share group. 

    7. How can an application query for robust buffer access support,
       since this is now determined at context creation time?

       RESOLVED. The application can query the value of ROBUST_ACCESS_EXT
       using GetIntegerv. If true, this functionality is enabled.

    8. How is the reset notification behavior controlled?

       RESOLVED: Reset notification behavior is determined at context
       creation time using EGL/GLX/WGL/etc. mechanisms. In order that shared
       objects be handled predictably, a context cannot share with
       another context unless both have the same reset notification
       behavior.


Revision History

    Rev.    Date       Author     Changes
    ----  ------------ ---------  ----------------------------------------
      3   31 Oct  2011 groth      Reverted to attribute for robust access. Now it's a
                                  companion to rather than subset of KHR_create_context
      2   11 Oct  2011 groth      Merged ANGLE and NV extensions.
                                  Convert to using flag to indicate robust access.
      1   15 July 2011 groth      Initial version
