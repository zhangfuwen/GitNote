# KHR_context_flush_control

Name

    KHR_context_flush_control

Name Strings

    GL_KHR_context_flush_control
    EGL_KHR_context_flush_control
    GLX_ARB_context_flush_control
    WGL_ARB_context_flush_control

Contact

    Graham Sellers (graham.sellers 'at' amd.com)

Contributors

    Graham Sellers, AMD
    Jon Leech

Notice

    Copyright (c) 2014 The Khronos Group Inc. Copyright terms at
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
    Approved by the ARB on June 26, 2014.
    Approved by the OpenGL ES Working Group on August 6, 2014.
    Ratified by the Khronos Board of Promoters on August 7, 2014 (non-EGL
    extensions, and on September 26, 2014 (EGL extension).

Version

    Last Modified Date:         9/28/2016
    Author Revision:            8

Number

    ARB Extension #168
    OpenGL ES Extension #191 (for GL_KHR_context_flush_control only)
    EGL Extension #102 (for EGL_KHR_context_flush_control only)

Dependencies

    GL_KHR_context_flush_control is written against the OpenGL 4.4 (Core
    Profile) specification, but can be implemented against any version of
    OpenGL or OpenGL ES.

    EGL_KHR_context_flush_control is written against the EGL 1.5
    Specification, but can be implemented against any version of EGL.

    GLX_ARB_context_flush_control is written against the GLX 1.4 and
    GLX_ARB_create_context specifications. Both are required.

    WGL_ARB_context_flush_control is written against the
    WGL_ARB_create_context specification, which is required.

Overview

    OpenGL and OpenGL ES have long supported multiple contexts. The
    semantics of switching contexts is generally left to window system
    binding APIs such as WGL, GLX and EGL. Most of these APIs (if not all)
    specify that when the current context for a thread is changed, the
    outgoing context performs an implicit flush of any commands that have
    been issued to that point. OpenGL and OpenGL ES define a flush as
    sending any pending commands for execution and that this action will
    result in their completion in finite time.

    This behavior has subtle consequences. For example, if an application is
    rendering to the front buffer and switches contexts, it may assume that
    any rendering performed thus far will eventually be visible to the user.
    With recent introduction of shared memory buffers, there become inumerable
    ways in which applications may observe side effects of an implicit flush
    (or lack thereof).

    In general, a full flush is not the desired behavior of the application.
    Nonetheless, applications that switch contexts frequently suffer the
    performance consequences of this unless implementations make special
    considerations for them, which is generally untenable.

    The EGL, GLX, and WGL extensions add new context creation parameters
    that allow an application to specify the behavior that is desired when a
    context is made non-current, and specifically to opt out of the implicit
    flush behavior. The GL extension allows querying the context flush
    behavior.

New Procedures and Functions

    None.

New Tokens (GL)

    NOTE: when implemented in an OpenGL ES context, all tokens defined by
    this extension must have a "_KHR" suffix. When implemented in an OpenGL
    context, all tokens must have NO suffix, as described below.

    Accepted by the <pname> parameter of GetIntegerv, GetFloatv, GetBooleanv
    GetDoublev and GetInteger64v:

        GL_CONTEXT_RELEASE_BEHAVIOR                         0x82FB

    Returned in <data> by GetIntegerv, GetFloatv, GetBooleanv
    GetDoublev and GetInteger64v when <pname> is
    GL_CONTEXT_RELEASE_BEHAVIOR:

        GL_NONE                                             0x0000
        GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH                   0x82FC

New Tokens (EGL)

    Accepted as an attribute name in the <*attrib_list> argument to
    eglCreateContext:

        EGL_CONTEXT_RELEASE_BEHAVIOR_KHR                    0x2097

    Accepted as an attribute value for EGL_CONTEXT_RELEASE_BEHAVIOR_KHR in
    the <*attrib_list> argument to eglCreateContext:

        EGL_CONTEXT_RELEASE_BEHAVIOR_NONE_KHR               0x0000
        EGL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_KHR              0x2098

New Tokens (GLX)

    Accepted as an attribute name in the <*attrib_list> argument to
    glXCreateContextAttribsARB:

        GLX_CONTEXT_RELEASE_BEHAVIOR_ARB                    0x2097

    Accepted as an attribute value for GLX_CONTEXT_RELEASE_BEHAVIOR_ARB in
    the <*attrib_list> argument to glXCreateContextAttribsARB:

        GLX_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB               0x0000
        GLX_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB              0x2098

New Tokens (WGL)

    Accepted as an attribute name in the <*attrib_list> argument to
    wglCreateContextAttribsARB:

        WGL_CONTEXT_RELEASE_BEHAVIOR_ARB                    0x2097

    Accepted as an attribute value for WGL_CONTEXT_RELEASE_BEHAVIOR_ARB in
    the <*attrib_list> argument to wglCreateContextAttribsARB:

        WGL_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB               0x0000
        WGL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB              0x2098

Additions to Chapter 22 of the OpenGL 4.4 Core Pofile Specification (Context
State Queries)

    In Subsection 22.2, "String Queries", after the description of
    CONTEXT_FLAGS, insert the following:

    The behavior of the context when it is made no longer current (released)
    by the platform binding may be queried by calling GetIntegerv with
    <value> CONTEXT_RELEASE_BEHAVIOR. If the behavior is
    CONTEXT_RELEASE_BEHAVIOR_FLUSH, any pending commands on the context will
    be flushed. If the behavior is NONE, pending commands are not flushed.

    The default value is CONTEXT_RELEASE_BEHAVIOR_FLUSH, and may in some
    cases be changed using platform-specific context creation extensions.


Additions to the EGL 1.5 Specification

    Add a new subsection to the description of eglCreateContext,
    following section 3.7.1.6 "OpenGL and OpenGL ES Reset Notification
    Strategy":

   "3.7.1.7 OpenGL and OpenGL ES Context Flush Behavior

    The attribute name EGL_CONTEXT_RELEASE_BEHAVIOR_KHR specifies the
    behavior of the rendering context when it is made non-current, as
    described in section 3.7.3. The attribute value may be one of
    EGL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_KHR or
    EGL_CONTEXT_RELEASE_BEHAVIOR_NONE_KHR. The default value is
    EGL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_KHR."


    Modify section 3.7.3 "Binding Contexts and Drawables", on p. 58:

    Replace the fourth paragraph in the description of eglMakeCurrent,
    starting "If the calling thread already has a current context...":

   "If the calling thread already has a current context of the same
    client API type as <ctx>, behavior is determined as follows:

    The current context is flushed if any of the following conditions
    are true:

    - The client API type of the current context is not OpenGL or OpenGL
      ES;
    - The client API type of the current context is OpenGL or OpenGL ES,
      and the context does not support the GL_KHR_context_flush_control
      extension
    - The client API type of the current context is OpenGL or OpenGL ES,
      the context supports the GL_KHR_context_flush_control extension,
      and the <release behavior> (the value of the
      EGL_CONTEXT_RELEASE_BEHAVIOR_KHR parameter) of the current context
      is EGL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_KHR.

    The only remaining case is treated differently. In this case, the client
    API type of the current context must be OpenGL or OpenGL ES; the context
    must support the GL_KHR_context_flush_control extension; and the release
    behavior must be EGL_CONTEXT_RELEASE_BEHAVIOR_NONE_KHR. In this case no
    action is taken, and it is as if the application simply stops making GL
    commands on the current context.

    After flushing (if appropriate) is performed, the current context is
    marked as no longer current. <ctx> is then made the current context
    for the calling thread. For purposes of ..."


Additions to WGL_ARB_create_context and wglMakeCurrent

    Add a new paragraph to the description of wglCreateContextAttribsARB, as
    defined by WGL_ARB_create_context:

    "The attribute name WGL_MAKE_NON_CONTEXT_BEHAVIOR_ARB specifies the
    behavior of the rendering context when it is made non-current. The
    attribute value may be one of WGL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB or
    WGL_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB. The default value of
    WGL_MAKE_NON_CONTEXT_BEHAVIOR_ARB is
    WGL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB."

    Define flushing behavior for wglMakeCurrent (there is no WGL
    specification; this language replaces the sentence "Before switching to
    the new rendering context, OpenGL flushes any previous rendering context
    that was current to the calling thread." in the Microsoft wglMakeCurrent
    reference page):

    "If the calling thread already has a current rendering context, behavior
    is determined by the <release behavior> (the value of the
    WGL_CONTEXT_RELEASE_BEHAVIOR_ARB parameter) of the current context. If
    the release behavior is WGL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB, pending
    commands to the previous context are flushed; if the release behavior is
    WGL_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB, then no action is taken, and it
    is as if the application simply stops making GL commands on that
    context. That context is marked as no longer current, and <ctx> is made
    the current context for the calling thread."

Additions to the GLX 1.4 Specification and GLX_ARB_create_context

    Add a new paragraph to the description of glXCreateContextAttribsARB,
    as defined by GLX_ARB_create_context:

    "The attribute name GLX_CONTEXT_RELEASE_BEHAVIOR_ARB specifies the
    behavior of the rendering context when it is made non-current. The
    attribute value may be one of GLX_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB or
    GLX_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB. The default value of
    GLX_CONTEXT_RELEASE_BEHAVIOR_ARB is GLX_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB."

    Modify section 3.3.7 "Rendering Contexts" on p. 27:

    Replace the third paragraph in the description of glXMakeContextCurrent,
    starting "If the calling thread already has a current rendering
    context...":

    "If the calling thread already has a current rendering context, behavior
    is determined by the <release behavior> (the value of the
    GLX_CONTEXT_RELEASE_BEHAVIOR_ARB parameter) of the current context. If
    the release behavior is GLX_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB, pending
    commands to the previous context are flushed; if the release behavior is
    GLX_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB, then no action is taken, and it
    is as if the application simply stops making GL commands on that
    context. That context is marked as no longer current, and <ctx> is made
    the current context for the calling thread."


GLX Protocol

    TBD.

Errors

    The errors defined for eglCreateContext, glXCreateContextAttribsARB, and
    wglCreateContextAttribsARB already describe the situations where invalid
    attribute values for the context release behavior are specified, and
    where no context can be created satisfying the specified behavior, so no
    new errors are added.

New State

    None.

New Implementation Dependent State

    Append to Table 23.54, "Implementation Dependent State" of the OpenGL 4.4 Specification:

    +-------------------------------+-------+--------------------+------------------------------------------+------------------------------------+---------+
    | Get Value                     | Type  | Get Command        | Initial Value                            | Description                        | Sec.    |
    +-------------------------------+-------+--------------------+------------------------------------------+------------------------------------+---------+
    | CONTEXT_RELEASE_BEHAVIOR      | E     | GetIntegerv        | See sec. 2.22                            | Specifies flush behavior when      | 22.2    |
    |                               |       |                    |                                          | the context is released.           |         |
    +-------------------------------+-------+--------------------+------------------------------------------+------------------------------------+---------+

Usage Examples

    TBD

Issues

    1) What happens when you switch between contexts that have the new
       CONTEXT_RELEASE_BEHAVIOR_NONE behavior?

       Nothing. It is as if the thread simply stopped calling functions on
       one context and started calling functions on another. If, for
       example, an application is rendering to multiple windows or multiple
       framebuffer objects, it may not care what order commands are executed
       across the contexts. This extension allows the application to rapidly
       switch between contexts that may or may not share objects without any
       implied flushes.

    2) Does this extension affect or change sharing semantics?

       RESOLVED: No. There is no intention to do so. From the client's
       perspective, at least, changes made to objects on one context should
       be visible to other contexts whenever they would have before.
       However, there may be some subtle interaction here that we haven't
       thought of yet.

    3) Do we need to be able to query the context behavior?

       RESOLVED. Yes. Added a state query for GL_CONTEXT_RELEASE_BEHAVIOR.

    4) Should EGL/GLX/WGL/GL enum values for similar tokens be shared?

       RESOLVED: the developing tradition is for window system binding enums
       for context creation to share the same values, but GL to have its own
       values lying in the traditional GL enum ranges.

    5) Should there be a GL extension string corresponding to
       the window system binding extension strings?

       RESOLVED: Yes. This is how we've always done it. Added
       GL_KHR_context_flush_control to Name Strings in revision 3.

    6) Should there be an EGL equivalent of this functionality? Should the
       GL extension be KHR or ARB?

       RESOLVED: Yes, there should be an EGL extension, but it will be
       approved on a different schedule by EGL and GL WGs. The GL extension
       should be KHR, since ES is also interested in this functionality.

    7) When the release behavior of an OpenGL compatibility profile context
       or an OpenGL ES 1.x context is to not flush, should it be possible to
       make a context non-current within a glBegin/glEnd pair?

       RESOLVED: No. Only GLX (not EGL or WGL) specifies this constraint,
       and hypothetical immediate-mode paths would have difficulty
       supporting MakeCurrent between Begin and End whether or not a flush
       is done.


Revision History

 Rev.   Date     Author    Changes
 ---- ---------  --------  ---------------------------------------------
   8  9/28/2016  Jon Leech Merge EGL_KHR_context_flush_control language,
                           which was ratified more than two years ago but
                           which we forgot to post until now.
   7  8/11/2014  Jon Leech Add issue 7, minor wording tweaks to include
                           GL and ES as peer APIs.
   6  8/06/2014  Jon Leech Change GL extension to KHR instead of ARB.
   5  6/26/2014  Jon Leech Remove ARB suffixes from GL tokens (only, Bug
                           12299) and clarify the initial value of the
                           GL flush behavior query.
   4  6/25/2014  Jon Leech Rewrite spec changes to be against the actual
                           GLX/WGL API and extension specification
                           documents, where possible.
   3  6/24/2014  Jon Leech Assign enum values, add issues 4/5.
   2  6/05/2014  gsellers  Add state query for retrieving context behavior.
                           Rename "lose-context" to "context release". This
                           is the terminology in the existing GLX spec. The
                           WGL spec doesn't appear to use a concrete term.
                           Update state table.
                           Renumbered issues to be contiguous.
   1  4/17/2014  gsellers  Initial revision based on earlier extension.
