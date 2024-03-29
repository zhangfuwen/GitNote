# NV_texture_barrier

Name

    NV_texture_barrier

Name Strings

    GL_NV_texture_barrier

Contact

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)

Contributors

    Mark Kilgard, NVIDIA
    Shazia Rahman, NVIDIA

Status

    Shipping (August 2009, Release 190)

Version

    Last Modified Date:         September 29, 2016
    NVIDIA Revision:            4

Number

    OpenGL Extension #381
    OpenGL ES Extension #271

Dependencies

    This extension is written against the OpenGL 3.0 specification.

    Also written based on the wording of the OpenGL ES 3.2 specification.

Overview

    This extension relaxes the restrictions on rendering to a currently
    bound texture and provides a mechanism to avoid read-after-write
    hazards.

New Procedures and Functions

    void TextureBarrierNV(void);

New Tokens

    None.

Additions to Chapter 2 of the OpenGL 3.0 Specification (OpenGL Operation)

    None.

Additions to Chapter 3 of the OpenGL 3.0 Specification (Rasterization)

    None.

Additions to Chapter 4 of the OpenGL 3.0 Specification (Per-Fragment
Operations and the Frame Buffer)

    Modify Section 4.4.3, Rendering When an Image of a Bound Texture Object
    is Also Attached to the Framebuffer, p. 288

    (Replace the complicated set of conditions with the following)

    Specifically, the values of rendered fragments are undefined if any 
    shader stage fetches texels and the same texels are written via fragment 
    shader outputs, even if the reads and writes are not in the same Draw 
    call, unless any of the following exceptions apply:

    - The reads and writes are from/to disjoint sets of texels (after 
      accounting for texture filtering rules).

    - There is only a single read and write of each texel, and the read is in 
      the fragment shader invocation that writes the same texel (e.g. using 
      "texelFetch2D(sampler, ivec2(gl_FragCoord.xy), 0);").

    - If a texel has been written, then in order to safely read the result
      a texel fetch must be in a subsequent Draw separated by the command
    
        void TextureBarrierNV(void);
        
      TextureBarrierNV() will guarantee that writes have completed and caches
      have been invalidated before subsequent Draws are executed.

Additions to Chapter 5 of the OpenGL 3.0 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL 3.0 Specification (State and
State Requests)

    None.

Additions to the AGL/GLX/WGL Specifications

    None

Additions to Chapter 9 of the OpenGL ES 3.2 Specification (Framebuffers
and Framebuffer Objects)

    Modify section 9.3.1, Rendering Feedback Loops:

    (Replace the complicated 2nd and 3rd paragraphs
    "Specifically... ...only be executed conditionally." with the
    following)

    Specifically, the values of rendered fragments are undefined if any
    shader stage fetches texels and the same texels are written via fragment
    shader outputs, even if the reads and writes are not in the same Draw
    call, unless any of the following exceptions apply:

    - The reads and writes are from/to disjoint sets of texels (after
      accounting for texture filtering rules).

    - There is only a single read and write of each texel, and the read is in
      the fragment shader invocation that writes the same texel (e.g. using
      "texelFetch2D(sampler, ivec2(gl_FragCoord.xy), 0);").

    - If a texel has been written, then in order to safely read the result
      a texel fetch must be in a subsequent Draw separated by the command

        void TextureBarrierNV(void);

      TextureBarrierNV() will guarantee that writes have completed and caches
      have been invalidated before subsequent Draws are executed.

Errors

New State

    None.

New Implementation Dependent State

    None.

GLX Protocol

    The following rendering command is sent to the server as
    a glXRender request:

    TextureBarrierNV

        2      4               rendering command length
        2      4348            rendering command opcode

Issues

    (1) What algorithms can take advantage of TextureBarrierNV?

      This can be used to accomplish a limited form of programmable blending
      for applications where a single Draw call does not self-intersect, by
      binding the same texture as both render target and texture and applying
      blending operations in the fragment shader. Additionally, bounding-box 
      optimizations can be used to minimize the number of TextureBarrierNV
      calls between Draws. For example:

        dirtybbox.empty();
        foreach (object in scene) {
          if (dirtybbox.intersects(object.bbox())) {
            TextureBarrierNV();
            dirtybbox.empty();
          }
          object.draw();
          dirtybbox = bound(dirtybbox, object.bbox());
        }

      Another application is to render-to-texture algorithms that ping-pong
      between two textures, using the result of one rendering pass as the input
      to the next. Existing mechanisms require expensive FBO Binds, DrawBuffer 
      changes, or FBO attachment changes to safely swap the render target and 
      texture. With texture barriers, layered geometry shader rendering, and 
      texture arrays, an application can very cheaply ping-pong between two 
      layers of a single texture. i.e.

        X = 0;
        // Bind the array texture to a texture unit
        // Attach the array texture to an FBO using FramebufferTexture3D
        while (!done) {
          // Stuff X in a constant, vertex attrib, etc.
          Draw - 
            Texturing from layer X;
            Writing gl_Layer = 1 - X in the geometry shader;
          
          TextureBarrierNV();
          X = 1 - X;
        }

      However, be warned that this requires geometry shaders and hence adds 
      the overhead that all geometry must pass through an additional program
      stage, so an application using large amounts of geometry could become 
      geometry-limited or more shader-limited.

    (2) Does this support OpenGL ES?

      RESOLVED:  Yes.  ES specification language has been added, written
      against the OpenGL 3.2 specification.  The added language is
      identical to the regular OpenGL language.

      As this specification has no dependencies other than assuming
      framebuffer objects, this extension could support any version of ES
      from 2.0 up.  However the texelFetch operation for fetching from a
      texture is introduced by OpenGL ES 3.0's GLSL or the NV_gpu_shader4
      extension.

Revision History

    Rev.    Date    Author    Changes
    ----  --------  --------  -----------------------------------------
     1              jbolz     Initial revision.
     2              mjk       Assign number.
     3              srahman   Add glx protocol specification.
     4    9/29/16   mjk       Add ES support
