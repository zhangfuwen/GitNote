# NV_timeline_semaphore

Name

    NV_timeline_semaphore

Name Strings

    GL_NV_timeline_semaphore

Contributors

    Carsten Rohde, NVIDIA
    James Jones, NVIDIA

Contact

    Carsten Rohde, NVIDIA Corporation (crohde 'at' nvidia.com)

Status

    Complete

Version

    Last Modified Date: Jul 10, 2020
    Revision:           1

Number

    551
    OpenGL ES Extension #330

Dependencies

    Written against the OpenGL 4.6 and OpenGL ES 3.2 specifications.

    GL_NV_timeline_semaphore requires GL_EXT_semaphore or a version of
    OpenGL or OpenGL ES that incorporates it.

Overview

    The Vulkan API introduces the concept of timeline semaphores.
    This extension brings those concepts to the OpenGL API by adding
    a semaphore type to the semaphore object. In OpenGL, timeline semaphore
    signal and wait operations are similar to the corresponding operations on
    imported Direct3D 12 fences defined in EXT_external_objects_win32.

New Procedures and Functions

    void CreateSemaphoresNV(sizei n, uint *semaphores);

    void SemaphoreParameterivNV(uint semaphore,
                                enum pname,
                                const GLint *params);

    void GetSemaphoreParameterivNV(uint semaphore,
                                   enum pname,
                                   int *params);

New Tokens

    Accepted by the <pname> parameter of SemaphoreParameterivNV
    and GetSemaphoreParameterivNV:

        SEMAPHORE_TYPE_NV                           0x95B3

    Accepted by the <param> parameter of SemaphoreParameterivNV and
    GetSemaphoreParameterivNV when <pname> parameter is SEMAPHORE_TYPE_NV:

        SEMAPHORE_TYPE_BINARY_NV                    0x95B4
        SEMAPHORE_TYPE_TIMELINE_NV                  0x95B5

    Accepted by the <pname> parameter of SemaphoreParameterui64vNV
    and GetSemaphoreParameterui64vNV:

        TIMELINE_SEMAPHORE_VALUE_NV                 0x9595

    Accepted by the <pname> parameter to GetIntegerv, GetFloatv, GetDoublev,
    GetInteger64v, and GetBooleanv:

        MAX_TIMELINE_SEMAPHORE_VALUE_DIFFERENCE_NV  0x95B6


Additions to Chapter 4 of the OpenGL 4.6 Specification (Event Model)

    Add the following to section 4.2 Semaphore Objects after paragraph
    which describes command GenSemaphoresEXT:

        The command

            void CreateSemaphoresNV(sizei n,
                                    uint *semaphores);

        returns <n> previously unused semaphore names in <semaphores>.
        The semaphores named contain default state, but initially have no
        external semaphores associated with them.

    Replace section 4.2.2 Semaphore Parameters with the following:

        Semaphore parameters control the type of the semaphore and how
        semaphore wait and signal operations behave.
        Table 4.3 defines which parameters are available for a semaphore
        based on the external handle type from which it was imported.
        Semaphore parameters are set using the commands

            void SemaphoreParameterivNV(uint semaphore,
                                        enum pname,
                                        const int *params);

        and

            void SemaphoreParameterui64vEXT(uint semaphore,
                                            enum pname,
                                            const uint64 *params);

        <semaphore> is the name of the semaphore object on which the
        parameter <pname> will be set to the value(s) in <pname>.

        Table 4.3: Semaphore parameters

        | Name                        | Handle Types                | Legal Values                         |
        +-----------------------------+-----------------------------+--------------------------------------+
        | SEMAPHORE_TYPE_NV           | any handle type             | SEMAPHORE_TYPE_BINARY_NV (default)   |
        |                             |                             | SEMAPHORE_TYPE_TIMELINE_NV           |
        +-----------------------------+-----------------------------+--------------------------------------+
        | TIMELINE_SEMAPHORE_VALUE_NV | any handle type             | any value                            |
        +-----------------------------+-----------------------------+--------------------------------------+

        The default type of a semaphore is SEMAPHORE_TYPE_BINARY_NV. Only when the semaphore is imported
        from a D3D fence, the semaphore type defaults to SEMAPHORE_TYPE_TIMELINE_NV.

        Parameters of a semaphore object may be queried with the commands

            void GetSemaphoreParameteriEXT(uint semaphore,
                                           enum pname,
                                           uint64 *params);

        and

            void GetSemaphoreParameterui64EXT(uint semaphore,
                                              enum pname,
                                              uint64 *params);

        <semaphore> is the semaphore object from with the parameter <pname>
        is queried.  The value(s) of the parameter are returned in <params>.
        <pname> may be any value in table 4.3.

    Add the following after the first paragraph of section 4.2.3 "Waiting
    for Semaphores"

        If <semaphore> is of the type SEMAPHORE_TYPE_TIMELINE_NV, it will
        reach the signaled state when its value is greater than or equal
        to the value specified by its TIMELINE_SEMAPHORE_VALUE_NV parameter.

    Add the following at the end of section 4.2.3 "Waiting for Semaphores":

        When using binary semaphores, for every wait on a semaphore there must
        be a prior signal of that semaphore that will not be consumed by a
        different wait on the semaphore.
        When using timeline semaphores, wait-before-signal behavior is
        well-defined and applications can wait for semaphore before the
        corresponding semaphore signal operation is flushed.

        MAX_TIMELINE_SEMAPHORE_VALUE_DIFFERENCE_NV indicates the maximum
        difference allowed by the implementation between the current value
        of a timeline semaphore and any pending wait operations

    Add the following after the first paragraph of section 4.2.4 "Signaling
    Semaphores"

        If <semaphore> is of the type SEMAPHORE_TYPE_TIMELINE_NV, its value
        will be set to the value specified by its TIMELINE_SEMAPHORE_VALUE_NV
        parameter when the signal operation completes.

    Add the following at the end of section 4.2.4 "Signaling for Semaphores":

        MAX_TIMELINE_SEMAPHORE_VALUE_DIFFERENCE_NV indicates the maximum
        difference allowed by the implementation between the current value
        of a timeline semaphore and any pending signal operations.


Example

    GLuint semapohre;
    glCreateSemaphoresNV(1, &semaphore);
    GLenum semaphoreType = GL_SEMAPHORE_TYPE_TIMELINE_NV;
    glSemaphoreParameterivNV(semaphore, GL_SEMAPHORE_TYPE_NV, (GLint*)&semaphoreType);
    glImportSemaphoreFdEXT(semaphore, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd); // or win32 equivalent

    GLuint64 semaphoreValue = 0;

    while (...) {
        glSemaphoreParameterui64vEXT(semaphore, GL_TIMELINE_SEMAPHORE_VALUE_NV, &semaphoreValue);
        glWaitSemaphoreEXT(semaphore, ...);

        ...

        semaphoreValue ++;
        glSemaphoreParameterui64vEXT(semaphore, GL_TIMELINE_SEMAPHORE_VALUE_NV, &semaphoreValue);
        glSignalSemaphoreEXT(semaphore, ...);
    }

    glDeleteSemaphoresEXT(1, &semaphore);


Issues

    (1) Should we add client functions to signal and wait for the semaphore on
        the CPU?

        RESOLVED: No. We already declined to add external Vulkan fence interop
                  with GL on the basis that you can just do that with Vulkan
                  if you need it.

    (2) Should GetIntegerv and GetBooleanv be allowed to query
        MAX_TIMELINE_SEMAPHORE_VALUE_DIFFERENCE_NV?

        RESOLVED: Yes. Although it's dangerous to use them they don't throw an
                  error but you are advised to use GetInteger64v.

Revision History

    Revision 1, 2020-07-10 (Carsten Rohde)
        - Initial draft.
