# NV_draw_vulkan_image

Name

    NV_draw_vulkan_image

Name Strings

    GL_NV_draw_vulkan_image

Contributors

    Jeff Bolz, NVIDIA Corporation

Contact

    Piers Daniell, NVIDIA Corporation (pdaniell 'at' nvidia.com)

Status

    Complete

Version

    Last Modified Date:         2/22/2017
    NVIDIA Revision:            2

Number

    OpenGL Extension #501
    OpenGL ES Extension #274

Dependencies

    This extension is written against the OpenGL 4.5 Specification
    (Compatibility Profile).

    This extension can also be used with OpenGL ES 3.2 or later.

    This extension interacts with Vulkan 1.0 and requires the OpenGL
    implementation to expose an implementation of Vulkan 1.0.

Overview

    This extension provides a new function, DrawVkImageNV(), allowing
    applications to draw a screen-aligned rectangle displaying some or all of
    the contents of a two-dimensional Vulkan VkImage.  Callers specify a
    Vulkan VkImage handle, an optional OpenGL sampler object, window
    coordinates of the rectangle to draw, and texture coordinates corresponding
    to the corners of the rectangle.  For each fragment produced by the
    rectangle, DrawVkImageNV  interpolates the texture coordinates, performs
    a texture lookup, and uses the texture result as the fragment color.

    No shaders are used by DrawVkImageNV; the results of the texture lookup
    are used in lieu of a fragment shader output.  The fragments generated are
    processed by all per-fragment operations.  In particular,
    DrawVkImageNV() fully supports blending and multisampling.

    In order to synchronize between Vulkan and OpenGL there are three other
    functions provided; WaitVkSemaphoreNV(), SignalVkSemaphoreNV() and
    SignalVkFenceNV().  These allow OpenGL to wait for Vulkan to complete work
    and also Vulkan to wait for OpenGL to complete work.  Together OpenGL
    and Vulkan can synchronize on the server without application
    interation.
    
    Finally the function GetVkProcAddrNV() is provided to allow the OpenGL
    context to query the Vulkan entry points directly and avoid having to
    load them through the typical Vulkan loader.

New Procedures and Functions

    void DrawVkImageNV(GLuint64 vkImage, GLuint sampler,
                       GLfloat x0, GLfloat y0, 
                       GLfloat x1, GLfloat y1,
                       GLfloat z,
                       GLfloat s0, GLfloat t0, 
                       GLfloat s1, GLfloat t1);
                       
    VULKANPROCNV GetVkProcAddrNV(const GLchar *name);
    
    void WaitVkSemaphoreNV (GLuint64 vkSemaphore);
    
    void SignalVkSemaphoreNV (GLuint64 vkSemaphore);
    
    void SignalVkFenceNV (GLuint64 vkFence);

New Types
    
    The Vulkan base entry point type from which all Vulkan functions pointers
    can be cast is:
    
        typedef void (APIENTRY *VULKANPROCNV)(void);

    Note that this function pointer is defined as having the
    same calling convention as the GL functions.

New Tokens

    None.

Additions to Chapter 1 of the OpenGL 4.5 Specification (Introduction)

    (Insert a new section after Section 1.3.6, OpenCL p. 7)
    
    1.3.X Vulkan
    
    Vulkan is a royalty-free, cross-platform explicit API for full-function
    3D graphics and compute.  Designed for a complete range of platforms from
    low-power mobile to high-performance desktop.
    
    OpenGL can interoperate directly with Vulkan to take advantage of Vulkan's
    explicit low-level access to the GPU for the power and performance
    efficiencies it can offet.
    
    An OpenGL application can use the following function to query the Vulkan
    function entry points from within an OpenGL context:
    
      VULKANPROCNV GetVkProcAddrNV(const GLchar *name);
      
    <name> is the name of the Vulkan function, for example "vkCreateInstance"
    and the return is a point to the Vulkan function address.  This allows
    OpenGL applications that need to interoperate with Vulkan to query the 
    entry points directly and bypass the typical Vulkan loader.  The OpenGL
    implementation provides access to the Vulkan implementation through this
    mechanism.
    
    The specification and more information about Vulkan can be found at
    https://www.khronos.org/vulkan/


Additions to Chapter 4 of the OpenGL 4.5 Specification (Event Model)

    (Insert a new section after Section 4.1.3, Sync Object Queries p. 42)
    
    4.1.X Synchronization between OpenGL and Vulkan
    
    The command:

      void WaitVkSemaphoreNV (GLuint64 vkSemaphore);
      
    causes the GL server to block until the Vulkan VkSemaphore <vkSemaphore>
    is signalled.  No GL commands after this command are executed by the server
    until the semaphore is signaled.  <vkSemaphore> must be a valid Vulkan
    VkSemaphore non-dispatchable handle otherwise the operation is undefined.
    
    The command:
    
      void SignalVkSemaphoreNV (GLuint64 vkSemaphore);
      
    causes the GL server to signal the Vulkan VkSemaphore <vkSemaphore> when
    it executes this command.  The semaphore is not signalled by GL until all
    commands issued before this have completed execution on the GL server.
    <vkSemaphore> must be a valid Vulkan VkSemaphore non-dispatchable handle
    otherwise the operation is undefined.
    
    The command:
    
      void SignalVkFenceNV (GLuint64 vkFence);
      
    causes the GL server to signal the Vulkan VkFence <vkFence> object when
    it executes this command.  The fence is not signalled by the GL until all
    commands issued before this have completed execution on the GL server.
    <vkFence> must be a valid Vulkan VkFence non-dispatcable handle otherwise
    the operation is undefined.

Additions to Chapter 10 of the OpenGL 4.5 Specification (Vertex Specification
and Drawing Commands)

    Modify Section 10.9, Conditional Rendering, p. 420

    (modify first paragraph to specify that DrawVkImageNV is affected by
    conditional rendering) ... is false, all rendering commands between
    BeginConditionalRender and the corresponding EndConditionalRender are
    discarded.  In this case, Begin, End, ...and DrawVkImageNV (section 18.4.X)
    have no effect.


Additions to Chapter 14 of the OpenGL 4.5 Specification (Fixed-Function
Primitive Assembly and Rasterization)

    Modify Section 14.1, Discarding Primitives Before Rasterization, p. 527

    (modify the end of the second paragraph) When enabled, RASTERIZER_DISCARD
    also causes the [[compatibility profile only:  Accum, Bitmap, CopyPixels,
    DrawPixels,]] Clear, ClearBuffer*, and DrawVkImageNV commands to be
    ignored.

Additions to Chapter 18 of the OpenGL 4.5 Specification (Drawing, Reading,
and Copying Pixels)

    (Insert new section after Section 18.4.1, Writing to the Stencil or
    Depth/Stencil Buffers, p. 621)

    Section 18.4.X, Drawing Textures

    The command:

      void DrawVkImageNV(GLuint64 vkImage, GLuint sampler,
                         GLfloat x0, GLfloat y0, 
                         GLfloat x1, GLfloat y1,
                         GLfloat z,
                         GLfloat s0, GLfloat t0, 
                         GLfloat s1, GLfloat t1);

    is used to draw a screen-aligned rectangle displaying a portion of the
    contents of the Vulkan image <vkImage>.  The four corners of this
    screen-aligned rectangle have the floating-point window coordinates
    (<x0>,<y0>), (<x0>,<y1>), (<x1>,<y1>), and (<x1>,<y0>).  A fragment will
    be generated for each pixel covered by the rectangle.  Coverage along the
    edges of the rectangle will be determined according to polygon
    rasterization rules.  If the framebuffer does not have a multisample
    buffer, or if MULTISAMPLE is disabled, fragments will be generated
    according to the polygon rasterization algorithm described in section
    3.6.1.  Otherwise, fragments will be generated for the rectangle using the
    multisample polygon rasterization algorithm described in section 3.6.6.
    In either case, the set of fragments generated is not affected by other
    state affecting polygon rasterization -- in particular, the CULL_FACE,
    POLYGON_SMOOTH, and POLYGON_OFFSET_FILL enables and PolygonMode state have
    no effect.  All fragments generated for the rectangle will have a Z window
    coordinate of <z>.

    The color associated with each fragment produced will be obtained by using
    an interpolated source coordinate (s,t) to perform a lookup into <vkImage>
    The (s,t) source coordinate for each fragment is interpolated over the
    rectangle in the manner described in section 3.6.1, where the (s,t)
    coordinates associated with the four corners of the rectangle are:

      (<s0>, <t0>) for the corner at (<x0>, <y0>),
      (<s1>, <t0>) for the corner at (<x1>, <y0>),
      (<s1>, <t1>) for the corner at (<x1>, <y1>), and
      (<s0>, <t1>) for the corner at (<x0>, <y1>).

    The interpolated texture coordinate (s,t) is used to obtain a texture
    color (Rs,Gs,Bs,As) from the <vkImage> using the process described in
    section 3.9.  The sampler state used for the texture access will be taken
    from the texture object <vkImage> if <sampler> is zero, or from the
    sampler object given by <sampler> otherwise.  The filtered texel <tau> is
    converted to an (Rb,Gb,Bb,Ab) vector according to table 3.25 and swizzled
    as described in Section 3.9.16.  [[Core Profile Only:  The section
    referenced here is present only in the compatibility profile; this
    language should be changed to reference the relevant language in the core
    profile.]]

    The fragments produced by the rectangle are not processed by fragment
    shaders [[Compatibility Profile:  or fixed-function texture, color sum, or
    fog operations]].  These fragments are processed by all of the
    per-fragment operations in section 4.1.  For the purposes of the scissor
    test (section 4.1.2), the enable and scissor rectangle for the first
    element in the array of scissor test enables and rectangles are used.

    The error INVALID_VALUE is generated by DrawVkImageNV if <sampler> is
    neither zero nor the name of a sampler object.  The error
    INVALID_OPERATION is generated if the image type of <vkImage> is not
    VK_IMAGE_TYPE_2D.


Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    TBD

Errors

    INVALID_VALUE is generated by DrawVkImageNV if <sampler> is neither
    zero nor the name of a sampler object.

    INVALID_OPERATION is generated by DrawVkImageNV if the target of <vkImage>
    is not VK_IMAGE_TYPE_2D.

New State

    None.


New Implementation Dependent State

    None.


Issues

    1) Can Vulkan entry points obtained through the typical Vulkan loader
       be used to interoperate with OpenGL.
       
       UNRESOLVED: Vulkan entry points obtained through the Vulkan loader may
       introduce layers between the application and the Vulkan driver.  These
       layers may modify the Vulkan non-dispatchable handles returned by the
       Vulkan driver.  In that case, these handles will not functions correctly
       when used with OpenGL interop.  It is therefore advised the Vulkan layers
       are bypassed when doing OpenGL interop by getting them directly from
       GetVkProcAddrNV().

Revision History
    
    Rev.    Date    Author    Changes
    ----  --------  --------  -----------------------------------------
     1    20160214  pdaniell  Initial draft
     2    20170222  pdaniell  Registered extension
