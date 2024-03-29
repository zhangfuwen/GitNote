# INTEL_framebuffer_CMAA

Name

    INTEL_framebuffer_CMAA

Name Strings

    GL_INTEL_framebuffer_CMAA

Contact

   Slawomir Grajewski,  Intel (slawomir.grajewski 'at' intel.com)

Contributors

    Devan Burke, Intel
    Miroslaw Oksiucik, Intel
    Jon Kennedy, Intel

Status

    Complete, shipping on selected Intel graphics.

Version

    Last Modified Date: June 26th, 2015
    Revision: 1

Number

    OpenGL Extension #481
    OpenGL ES Extension #246

Dependencies

    OpenGL dependencies:

        OpenGL 3.0 is required.

        The extension is written against the OpenGL 4.5 Specification, Core
        Profile, October 18, 2013.

    OpenGL ES dependencies:

        This extension is written against the OpenGL ES 3.1 specification,
        June 4, 2014.

Overview

    Multisampling is a mechanism to antialias all GL primitives and is part of
    the GL specification.

    Better visual quality can be achieved by applying multisampling. However,
    on certain platforms it comes at a high performance cost. In general, the
    greater number of samples per pixel, the bigger the cost.

    Conservative Morphological Anti-Aliasing (CMAA) is an alternative approach
    to antialiasing, which operates on the final image. This post processing
    technique results in image quality comparable to multisampling at much
    lower cost and better performance.

    This extension incorporates an optimized CMAA algorithm implementation into
    the GL implementation.

    For more information on CMAA refer to http://software.intel.com.

New Procedures and Functions

    void ApplyFramebufferAttachmentCMAAINTEL();

New Tokens

    None.

Add new Section 17.4.6 to Chapter 17.4, Whole Framebuffer Operations, OpenGL 4.5

    17.4.6 Framebuffer post processing: Conservative Morphological Anti-Aliasting (CMMA)

Add new Section 15.2.5 to Chapter 15.2, Whole Framebuffer Operations, OpenGL ES 3.1

    15.2.5 Framebuffer post processing: Conservative Morphological Anti-Aliasting (CMMA)

    The post processing of current draw buffers pertaining to the framebuffer
    bound to the DRAW_FRAMEBUFFER target according to the Conservative
    Morphological Anti-Aliasing algorithm can be requested by calling

    void ApplyFramebufferAttachmentCMAAINTEL();

    Errors

        An INVALID_OPERATION error is generated if any of the current draw
        buffers is multisampled.

        An OUT_OF_MEMORY error may be generated if the GL implementation is
        unable to allocate internal resources necessary to perform the
        requested operation.

    The CMAA post processing is performed in-place for all color attachments of
    the currently bound draw framebuffer.

Issues

    1. If an application wants to post process color attachments of different
    dimensions/internal formats, what is the best approach in terms of
    performance?

       a. Attach the color attachment to the same framebuffer one after another
          for post processing?

       b. For each color attachment create a dedicated framebuffer object that
          will be used for post processing?

    RESOLVED: b.

    Whilst both approaches are functionally correct, approach b is more
    performant as the GL implementation will allocate temporary resources per
    framebuffer object and will reuse those resources for subsequent post
    processing operations without the need to resize or re-allocate them.

    2. Does the application have to perform any synchronization after requesting
       CMAA post processing?

    RESOLVED: No.

    From synchronization perspective, the post processing request behaves
    exactly as any draw request.

    3. What happens if CMAA post processing is requested for the multisampled
       color attachment?

    RESOLVED

    An INVALID_OPERATION error code is returned.

Revision History

    1.2   08/04/15 sgrajewski   Minor clarifications.
    1.1   06/26/15 sgrajewski   Initial revision.

