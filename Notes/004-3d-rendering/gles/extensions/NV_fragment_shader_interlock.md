# NV_fragment_shader_interlock

Name

    NV_fragment_shader_interlock

Name Strings

    GL_NV_fragment_shader_interlock

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA Corporation
    Mathias Heyer, NVIDIA Corporation

Status

    Shipping

Version

    Last Modified Date:         March 27, 2015
    NVIDIA Revision:            2

Number

    OpenGL Extension #468
    OpenGL ES Extension #230

Dependencies

    This extension is written against the OpenGL 4.3
    (Compatibility Profile, dated February 14, 2013), and the
    OpenGL ES 3.1.0 (dated March 17, 2014) Specification

    This extension is written against the OpenGL Shading Language
    Specification (version 4.30, revision 8) and the OpenGL ES Shading
    Language Specification (version 3.10, revision 2).

    OpenGL 4.3 and GLSL 4.30 are required in an OpenGL implementation
    OpenGL ES 3.1 and GLSL ES 3.10 are required in an OpenGL ES implementation

    This extension interacts with NV_shader_buffer_load and
    NV_shader_buffer_store.

    This extension interacts with NV_gpu_program4 and NV_gpu_program5.

    This extension interacts with EXT_tessellation_shader.

    This extension interacts with OES_sample_shading

    This extension interacts with OES_shader_multisample_interpolation

    This extension interacts with OES_shader_image_atomic

Overview

    In unextended OpenGL 4.3 or OpenGL ES 3.1, applications may produce a
    large number of fragment shader invocations that perform loads and
    stores to memory using image uniforms, atomic counter uniforms,
    buffer variables, or pointers. The order in which loads and stores
    to common addresses are performed by different fragment shader
    invocations is largely undefined.  For algorithms that use shader
    writes and touch the same pixels more than once, one or more of the
    following techniques may be required to ensure proper execution ordering:

      * inserting Finish or WaitSync commands to drain the pipeline between
        different "passes" or "layers";

      * using only atomic memory operations to write to shader memory (which
        may be relatively slow and limits how memory may be updated); or

      * injecting spin loops into shaders to prevent multiple shader
        invocations from touching the same memory concurrently.

    This extension provides new GLSL built-in functions
    beginInvocationInterlockNV() and endInvocationInterlockNV() that delimit a
    critical section of fragment shader code.  For pairs of shader invocations
    with "overlapping" coverage in a given pixel, the OpenGL implementation
    will guarantee that the critical section of the fragment shader will be
    executed for only one fragment at a time.

    There are four different interlock modes supported by this extension,
    which are identified by layout qualifiers.  The qualifiers
    "pixel_interlock_ordered" and "pixel_interlock_unordered" provides mutual
    exclusion in the critical section for any pair of fragments corresponding
    to the same pixel.  When using multisampling, the qualifiers
    "sample_interlock_ordered" and "sample_interlock_unordered" only provide
    mutual exclusion for pairs of fragments that both cover at least one
    common sample in the same pixel; these are recommended for performance if
    shaders use per-sample data structures.

    Additionally, when the "pixel_interlock_ordered" or
    "sample_interlock_ordered" layout qualifier is used, the interlock also
    guarantees that the critical section for multiple shader invocations with
    "overlapping" coverage will be executed in the order in which the
    primitives were processed by the GL.  Such a guarantee is useful for
    applications like blending in the fragment shader, where an application
    requires that fragment values to be composited in the framebuffer in
    primitive order.

    This extension can be useful for algorithms that need to access per-pixel
    data structures via shader loads and stores.  Such algorithms using this
    extension can access such data structures in the critical section without
    worrying about other invocations for the same pixel accessing the data
    structures concurrently.  Additionally, the ordering guarantees are useful
    for cases where the API ordering of fragments is meaningful.  For example,
    applications may be able to execute programmable blending operations in
    the fragment shader, where the destination buffer is read via image loads
    and the final value is written via image stores.

New Procedures and Functions

    None.

New Tokens

    None.

Modifications to the OpenGL 4.3 Specification (Compatibility Profile)

    None.

Modifications to the OpenGL Shading Language Specification, Version 4.30

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_NV_fragment_shader_interlock : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_NV_fragment_shader_interlock           1


    Modify Section 4.4.1.3, Fragment Shader Inputs (p. 58)

    (add to the list of layout qualifiers containing "early_fragment_tests",
     p. 59, and modify the surrounding language to reflect that multiple
     layout qualifiers are supported on "in")

      layout-qualifier-id
        pixel_interlock_ordered
        pixel_interlock_unordered
        sample_interlock_ordered
        sample_interlock_unordered

    (add to the end of the section, p. 59)

    The identifiers "pixel_interlock_ordered", "pixel_interlock_unordered",
    "sample_interlock_ordered", and "sample_interlock_unordered" control the
    ordering of the execution of shader invocations between calls to the
    built-in functions beginInvocationInterlockNV() and
    endInvocationInterlockNV(), as described in section 8.13.3. A
    compile or link error will be generated if more than one of these layout
    qualifiers is specified in shader code. If a program containing a
    fragment shader includes none of these layout qualifiers, it is as
    though "pixel_interlock_ordered" were specified.

    Add to the end of Section 8.13, Fragment Processing Functions (p. 168)

    8.13.3, Fragment Shader Execution Ordering Functions

    By default, fragment shader invocations are generally executed in
    undefined order. Multiple fragment shader invocations may be executed
    concurrently, including multiple invocations corresponding to a single
    pixel. Additionally, fragment shader invocations for a single pixel might
    not be processed in the order in which the primitives generating the
    fragments were specified in the OpenGL API.

    The paired functions beginInvocationInterlockNV() and
    endInvocationInterlockNV() allow shaders to specify a critical section,
    inside which stronger execution ordering is guaranteed.  When using the
    "pixel_interlock_ordered" or "pixel_interlock_unordered" qualifier,
    ordering guarantees are provided for any pair of fragment shader
    invocations X and Y triggered by fragments A and B corresponding to the
    same pixel. When using the "sample_interlock_ordered" or
    "sample_interlock_unordered" qualifier, ordering guarantees are provided
    for any pair of fragment shader invocations X and Y triggered by fragments
    A and B that correspond to the same pixel, where at least one sample of
    the pixel is covered by both fragments. No ordering guarantees are
    provided for pairs of fragment shader invocations corresponding to
    different pixels. Additionally, no ordering guarantees are provided for
    pairs of fragment shader invocations corresponding to the same fragment.
    When multisampling is enabled and the framebuffer has sample buffers,
    multiple fragment shader invocations may result from a single fragment due
    to the use of the "sample" auxilliary storage qualifier, OpenGL API
    commands forcing multiple shader invocations per fragment, or for other
    implementation-dependent reasons.

    When using the "pixel_interlock_unordered" or "sample_interlock_unordered"
    qualifier, the interlock will ensure that the critical sections of
    fragment shader invocations X and Y with overlapping coverage will never
    execute concurrently. That is, invocation X is guaranteed to complete its
    call to endInvocationInterlockNV() before invocation Y completes its call
    to beginInvocationInterlockNV(), or vice versa.

    When using the "pixel_interlock_ordered" or "sample_interlock_ordered"
    layout qualifier, the critical sections of invocations X and Y with
    overlapping coverage will be executed in a specific order, based on the
    relative order assigned to their fragments A and B.  If fragment A is
    considered to precede fragment B, the critical section of invocation X is
    guaranteed to complete before the critical section of invocation Y begins.
    When a pair of fragments A and B have overlapping coverage, fragment A is
    considered to precede fragment B if

      * the OpenGL API command producing fragment A was called prior to the
        command producing B, or

      * the point, line, triangle, [[compatibility profile: quadrilateral,
        polygon,]] or patch primitive producing fragment A appears earlier in
        the same strip, loop, fan, or independent primitive list producing
        fragment B.

    When [[compatibility profile: decomposing quadrilateral or polygon
    primitives or]] tessellating a single patch primitive, multiple
    primitives may be generated in an undefined implementation-dependent
    order.  When fragments A and B are generated from such unordered
    primitives, their ordering is also implementation-dependent.

    If fragment shader X completes its critical section before fragment shader
    Y begins its critical section, all stores to memory performed in the
    critical section of invocation X using a pointer, image uniform, atomic
    counter uniform, or buffer variable qualified by "coherent" are guaranteed
    to be visible to any reads of the same types of variable performed in the
    critical section of invocation Y.

    If multisampling is disabled, or if the framebuffer does not include
    sample buffers, fragment coverage is computed per-pixel. In this case,
    the "sample_interlock_ordered" or "sample_interlock_unordered" layout
    qualifiers are treated as "pixel_interlock_ordered" or
    "pixel_interlock_unordered", respectively.


      Syntax:

        void beginInvocationInterlockNV(void);
        void endInvocationInterlockNV(void);

      Description:

    The beginInvocationInterlockNV() and endInvocationInterlockNV() may only
    be placed inside the function main() of a fragment shader and may not be
    called within any flow control.  These functions may not be called after a
    return statement in the function main(), but may be called after a discard
    statement.  A compile- or link-time error will be generated if main()
    calls either function more than once, contains a call to one function
    without a matching call to the other, or calls endInvocationInterlockNV()
    before calling beginInvocationInterlockNV().

Additions to the AGL/GLX/WGL Specifications

    None.

Errors

    None.

New State

    None.

New Implementation Dependent State

    None.

Interactions with OpenGL ES 3.1

    Disabling multisample rasterization is not available on OpenGL ES;
    it is always enabled.


Dependencies on EXT_tessellation_shader

     If this extension is implemented on OpenGL ES and EXT_tessellation_shader
     is not supported, remove language referring to tessellation of patch
     primitives.


Dependencies on OES_sample_shading

     If this extension is implemented on OpenGL ES and OES_sample_shading
     is not supported, remove references to per-sample shading via
     MinSampleShading[OES]()


Dependencies on OES_shader_image_atomic

    If this extension is implemented on OpenGL ES and OES_shader_image_atomic
    is not supported, disregard language referring to atomic memory operations.


Dependencies on OES_shader_multisample_interpolation

   If this extension is implemented on OpenGL ES and OES_shader_-
   multisample_interpolation is not supported, ignore language
   about the "sample" auxilliary storage qualifier.


Dependencies on NV_shader_buffer_load and NV_shader_buffer_store

    If NV_shader_buffer_load and NV_shader_buffer_store are not supported,
    references to ordering memory accesses using pointers should be deleted.


Dependencies on NV_gpu_program4 and NV_fragment_program4

    Modify Section 2.X.2, Program Grammar, of the NV_fragment_program4
    specification (which modifies the NV_gpu_program4 base grammar)

      <SpecialInstruction>    ::= "FSIB"
                                | "FSIE"


    Modify Section 2.X.4, Program Execution Environment

    (add to the opcode table)

                  Modifiers
      Instruction F I C S H D  Out Inputs    Description
      ----------- - - - - - -  --- --------  --------------------------------
      FSIB        - - - - - -  -   -         begin fragment shader interlock
      FSIE        - - - - - -  -   -         end fragment shader interlock


    Modify Section 2.X.6, Program Options

    + Fragment Shader Interlock (NV_pixel_interlock_ordered,
      NV_pixel_interlock_unordered, NV_sample_interlock_ordered, and
      NV_sample_interlock_ordered)

    If a fragment program specifies the "NV_pixel_interlock_ordered",
    "NV_pixel_interlock_unordered", "NV_sample_interlock_ordered", or
    "NV_sample_interlock_ordered" options, it will configure a critical
    section using the FSIB (fragment shader interlock begin) and FSIE opcodes
    (fragment shader interlock end) opcodes.  The execution of the critical
    sections will be ordered for pairs of program invocations corresponding to
    the same pixel, as described in Section 8.13.3 of the OpenGL Shading
    Language Specification, where the four options are considered to specify
    layout qualifiers with names equivalent to matching the program option.

    A program will fail to load if it specifies more than one of these program
    options, if it specifies exactly one of these options but does not contain
    exactly one FSIB instruction and one FSIE instruction, or if it contains
    an FSIB or FSIE instruction without specifying any of these options.


    Add the following subsections to section 2.X.8, Program Instruction Set


    Section 2.X.8.Z, FSIB:  Fragment Shader Interlock Begin

    The FSIB instruction specifies the beginning of a critical section in a
    fragment program, where execution of the critical section is ordered
    relative to other fragments.  This instruction has no other effect.

    The FSIB instruction is not allowed in arbitrary locations in a program.
    A program will fail to load if it includes an FSIB instruction inside a
    IF/ELSE/ENDIF block, inside a REP/ENDREP block, or inside any subroutine
    block other than the one labeled "main".  Additionally, a program will
    fail to load if it contains more than one FSIB instruction, or if its one
    FSIB instruction is not followed by an FSIE instruction.

    FSIB has no operands and generates no result.


    Section 2.X.8.Z, FSIE:  Fragment Shader Interlock End

    The FSIE instruction specifies the end of a critical section in a fragment
    program, where execution of the critical section is ordered relative to
    other fragments.  This instruction has no other effect.

    The FSIE instruction is not allowed in arbitrary locations in a program.
    A program will fail to load if it includes an FSIE instruction inside a
    IF/ELSE/ENDIF block, inside a REP/ENDREP block, or inside any subroutine
    block other than the one labeled "main".  Additionally, a program will
    fail to load if it contains more than one FSIE instruction, or if its one
    FSIE instruction is not preceded by an FSIB instruction.

    FSIE has no operands and generates no result.

Issues

    (1) What should this extension be called?

      RESOLVED:  NV_fragment_shader_interlock.  The
      beginInvocationInterlockNV() and endInvocationInterlockNV() commands
      identify a critical section during which other invocations with
      overlapping coverage are locked out until the critical section
      completes.

    (2) When using multisampling, the OpenGL specification permits
        multiple fragment shader invocations to be generated for a single
        fragment.  For example, per-sample shading using the "sample"
        auxilliary storage qualifier or the MinSampleShading() OpenGL API command
        can be used to force per-sample shading.  What execution ordering
        guarantees are provided between fragment shader invocations generated
        from the same fragment?

      RESOLVED:  We don't provide any ordering guarantees in this extension.
      This implies that when using multisampling, there is no guarantee that
      two fragment shader invocations for the same fragment won't be executing
      their critical sections concurrently.  This could cause problems for
      algorithms sharing data structures between all the samples of a pixel
      unless accesses to these data structures are performed atomically.

      When using per-sample shading, the interlock we provide *does* guarantee
      that no two invocations corresponding to the same sample execute the
      critical section concurrently.  If a separate set of data structures is
      provided for each sample, no conflicts should occur within the critical
      section.

      Note that in addition to the per-sample shading options in the shading
      language and API, implementations may provide multisample antialiasing
      modes where the implementation can't simply run the fragment shader once
      and broadcast results to a large set of covered samples.

    (3) What performance differences are expected between shaders using the
       "pixel" and "sample" layout qualifier variants in this extension (e.g.,
       "pixel_invocation_ordered" and "sample_invocation_ordered")?

      RESOLVED:  We expect that shaders using "sample" qualifiers may have
      higher performance, since the implementation need not order pairs of
      fragments that touch the same pixel with "complementary" coverage.  Such
      situations are fairly common:  when two adjacent triangles combine to
      cover a given pixel, two fragments will be generated for the pixel but
      no sample will be covered by both.  When using "sample" qualifiers, the
      invocations for both fragments can run concurrently.  When using "pixel"
      qualifiers, the critical section for one fragment must wait until the
      critical section for the other fragment completes.

    (4) What performance differences are expected between shaders using the
       "ordered" and "unordered" layout qualifier variants in this extension
       (e.g., "pixel_invocation_ordered" and "pixel_invocation_unordered")?

      RESOLVED:  We expect that shaders using "unordered" may have higher
      performance, since the critical section implementation doesn't need to
      ensure that all previous invocations with overlapping coverage have
      completed their critical sections.  Some algorithms (e.g., building data
      structures in order-independent transparency algorithms) will require
      mutual exclusion when updating per-pixel data structures, but do not
      require that shaders execute in a specific ordering.

    (5) Are fragment shaders using this extension allowed to write outputs?
        If so, is there any guarantee on the order in which such outputs are
        written to the framebuffer?

      RESOLVED:  Yes, fragment shaders with critical sections may still write
      outputs.  If fragment shader outputs are written, they are stored or
      blended into the framebuffer in API order, as is the case for fragment
      shaders not using this extension.

    (6) What considerations apply when using this extension to implement a
        programmable form of conventional blending using image stores?

      RESOLVED:  Per-fragment operations performed in the pipeline following
      fragment shader execution obviously have no effect on image stores
      executing during fragment shader execution.  In particular, multisample
      operations such as broadcasting a single fragment output to multiple
      samples or modifying the coverage with alpha-to-coverage or a shader
      coverage mask output value have no effect.  Fragments can not be killed
      before fragment shader blending using the fixed-function alpha test or
      using the depth test with a Z value produced by the shader.  Fragments
      will normally not be killed by fixed-function depth or stencil tests,
      but those tests can be enabled before fragment shader invocations using
      the layout qualifier "early_fragment_tests".  Any required
      fixed-function features that need to be handled before programmable
      blending that aren't enabled by "early_fragment_tests" would need to be
      emulated in the shader.

      Note also that performing blend computations in the shader are not
      guaranteed to produce results that are bit-identical to these produced
      by fixed-function blending hardware, even if mathematically equivalent
      algorithms are used.

    (7) For operations accessing shared per-pixel data structures in the
        critical section, what operations (if any) must be performed in shader
        code to ensure that stores from one shader invocation are visible to
        the next?

      RESOLVED:  The "coherent" qualifier is required in the declaration of
      the shared data structures to ensure that writes performed by one
      invocation are visible to reads performed by another invocation.

      In shaders that don't use the interlock, "coherent" is not sufficient as
      there is no guarantee of the ordering of fragment shader invocations --
      even if invocation A can see the values written by another invocation B,
      there is no general guarantee that invocation A's read will be performed
      before invocation B's write.  The built-in function memoryBarrier() can
      be used to generate a weak ordering by which threads can communicate,
      but it doesn't order memory transactions between two separate
      invocations.  With the interlock, execution ordering between two threads
      from the same pixel is well-defined as long as the loads and stores are
      performed inside the critical section, and the use of "coherent" ensures
      that stores done by one invocation are visible to other invocations.

    (8) Should we provide an explicit mechanisms for shaders to indicate a
        critical section?  Or should we just automatically infer a critical
        section by analyzing shader code?  Or should we just wrap the entire
        fragment shader in a critical section?

      RESOLVED:  Provide an explicit critical section.

      We definitely don't want to wrap the entire shader in a critical section
      when a smaller section will suffice.  Doing so would hold off the
      execution of any other fragment shader invocation with the same (x,y)
      for the entire (potentially long) life of the fragment shader.  Hardware
      would need to track a large number of fragments awaiting execution, and
      may be so backed up that further fragments will be blocked even if they
      don't overlap with any fragments currently executing.  Providing a
      smaller critical section reduces the amount of time other fragments are
      blocked and allows implementations to perform useful work for
      conflicting fragments before they hit the critical section.

      While a compiler could analyze the code and wrap a critical section
      around all memory accesses, it may be difficult to determine which
      accesses actually require mutual exclusion and ordering, and which
      accesses are safe to do with no protection.  Requiring shaders to
      explicitly identify a critical section doesn't seem overwhelmingly
      burdensome, and allows applications to exclude memory accesses that it
      knows to be "safe".

    (9) What restrictions should be imposed on the use of the
        beginInvocationInterlockNV() and endInvocationInterlockNV() functions
        delimiting a critical section?

      RESOLVED:  We impose restrictions similar to those on the barrier()
      built-in function in tessellation control shaders to ensure that any
      shader using this functionality has a single critical section that can
      be easily identified during compilation.  In particular, we require that
      these functions be called in main() and don't permit them to be called
      in conditional flow control.

      These restrictions ensure that there is always exactly one call to the
      "begin" and "end" functions in a predictable location in the compiled
      shader code, and ensure that the compiler and hardware don't have to
      deal with unusual cases (like entering a critical section and never
      leaving, leaving a critical section without entering it, or trying to
      enter a critical section more than once).

Revision History

    Revision 2, 2015/03/27
      - Add ES interactions

    Revision 1
      - Internal revisions
