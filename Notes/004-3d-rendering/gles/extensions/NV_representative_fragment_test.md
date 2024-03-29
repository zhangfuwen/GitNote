# NV_representative_fragment_test

Name

    NV_representative_fragment_test

Name Strings

    GL_NV_representative_fragment_test

Contacts

    Christoph Kubisch, NVIDIA Corporation (ckubisch 'at' nvidia.com)
    Kedarnath Thangudu, NVIDIA Corporation (kthangudu 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA Corporation
    Pat Brown, NVIDIA Corporation
    Eric Werness, NVIDIA Corporation
    Pyarelal Knowles, NVIDIA Corporation

Status

    Shipping

Version

    Last Modified Date:     March 7, 2019
    NVIDIA Revision:        3

Number

    OpenGL Extension #528
    OpenGL ES Extension #314

Dependencies

    This extension is written against the OpenGL 4.6 Specification
    (Compatibility Profile), dated May 14, 2018.

    OpenGL 4.5 or OpenGL ES 3.2 is required.

Overview

    This extension provides a new _representative fragment test_ that allows
    implementations to reduce the amount of rasterization and fragment
    processing work performed for each point, line, or triangle primitive. For
    any primitive that produces one or more fragments that pass all other
    early fragment tests, the implementation is permitted to choose one or
    more "representative" fragments for processing and discard all other
    fragments. For draw calls rendering multiple points, lines, or triangles
    arranged in lists, strips, or fans, the representative fragment test is
    performed independently for each of those primitives.

    This extension is useful for applications that use an early render pass
    to determine the full set of primitives that would be visible in the final
    scene. In this render pass, such applications would set up a fragment
    shader that enables early fragment tests and writes to an image or shader
    storage buffer to record the ID of the primitive that generated the
    fragment. Without this extension, the shader would record the ID
    separately for each visible fragment of each primitive. With this
    extension, fewer stores will be performed, particularly for large
    primitives.

    The representative fragment test has no effect if early fragment tests are
    not enabled via the fragment shader. The set of fragments discarded by the
    representative fragment test is implementation-dependent and may vary from
    frame to frame. In some cases, the representative fragment test may not
    discard any fragments for a given primitive.


New Procedures and Functions

    None

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, and IsEnabled,
    and by the <pname> parameter of GetBooleanv, GetIntegerv,
    GetFloatv, and GetDoublev:

        REPRESENTATIVE_FRAGMENT_TEST_NV       0x937F

Modifications to the OpenGL 4.6 Specification (Compatibility Profile)

    Modify Section 14.9, Early Per-Fragment Tests (p. 578)

    (modify second pararaph of the section, p. 578, to document that there are
     now four optional early fragment tests)

    Three fragment operations are performed, and a further four are
    optionally performed on each fragment, ...

    (modify the last paragraph, p. 578, to list the new early fragment test)

    If early per-fragment operations are enabled, these tests are also
    performed:

        * the stencil test (see section 17.3.3);
        * the depth buffer test (see section 17.3.4); and
        * the representative fragment test (see section 17.3.X)
        * occlusion query sample counting (see section 17.3.5)


    Modify Section 14.9.4, The Early Fragment Test Qualifier, p. 582

    (modify the first paragraph of the section, p. 582, to enumerate the new
     test)

    The stencil test, depth buffer test, representative fragment test, and
    occlusion query sample counting are performed if and only if early
    fragment tests are enabled in the active fragment shader (see section
    15.2.4). ...


    Insert new section before Section 17.3.5, Occlusion Queries (p. 614)

    Section 17.3.X, Representative Fragment Test

    The representative fragment test allows implementations to reduce the
    amount of rasterization and fragment processing work performed for each
    point, line, or triangle primitive. For any primitive that produces one or
    more fragments that pass all prior early fragment tests, the
    implementation is permitted to choose one or more "representative"
    fragments for processing and discard all other fragments. For draw calls
    rendering multiple points, lines, or triangles arranged in lists, strips,
    or fans, the representative fragment test is performed independently for
    each of those primitives. The set of fragments discarded by the
    representative fragment test is implementation-dependent. In some cases,
    the representative fragment test may not discard any fragments for a given
    primitive.

    This test is enabled or disabled using Enable or Disable with the target
    REPRESENTATIVE_FRAGMENT_NV. If early fragment tests (section 15.2.4) are
    not enabled in the active fragment shader, the representative fragment
    test has no effect, even if enabled.


Additions to the AGL/GLX/WGL Specifications

    None.

New State

    Get Value                              Type    Get Command   Initial Value   Description                Sec    Attribute
    ------------------------------------   ----    -----------   -------------   -------------------------  ------ --------------
    REPRESENTATIVE_FRAGMENT_TEST_NV          B      IsEnabled     GL_FALSE       Representative fragment    17.3.X enable
                                                                                 test

New Implementation Dependent State

    None

Interactions with OpenGL ES

    If implemented with OpenGL ES, ignore references to GetDoublev.

Issues

    (1) Since the representative fragment test does not have guaranteed
        behavior, it is sort of a hint.  Should we use the existing hint
        mechanisms for this extension or simply add an enable?

    RESOLVED:  Use an enable.  Hints are rarely used in OpenGL, and the
    "FASTEST" vs. "NICEST" vs. "DONT_CARE" doesn't map reasonably to the
    representative fragment test.

    (2) Should this functionality be exposed as a sub-feature of the depth or
        stencil tests, as its own separate per-fragment test, or as some piece
        of state controlling primitive rasterization?

    RESOLVED:  Expose as a per-fragment test.  This test is largely orthogonal
    to depth testing, other than it is supposed to run after the depth
    testing.  So coupling it to the depth test doesn't make sense.  Coupling
    the feature to rasterization also doesn't make too much sense, because the
    rasterization pipeline stage discarding fragments for this test would
    depend on a later pipeline stages performing other per-fragment tests
    (such as the depth test).


Revision History

    Revision 3, March 7, 2019 (pknowles)
    - Add ES interactions.

    Revision 2, September 15, 2018 (pbrown)
    - Prepare specification for publication.

    Revision 1 (ckubisch and kthangudu)
    - Internal Revisions
