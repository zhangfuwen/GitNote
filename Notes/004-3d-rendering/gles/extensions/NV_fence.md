# NV_fence

Name

    NV_fence

Name Strings

    GL_NV_fence

Contact

    John Spitzer, NVIDIA Corporation (jspitzer 'at' nvidia.com)
    Mark Kilgard, NVIDIA Corporation (mjk 'at' nvidia.com)

Contributors

    John Spitzer
    Mark Kilgard
    Acorn Pooley

Notice

    Copyright NVIDIA Corporation, 2000, 2001.

IP Status

    NVIDIA Proprietary.

Status

    Shipping as of June 8, 2000 (version 1.0)

    Shipping as of November, 2003 (version 1.1)

    Version 1.2 adds ES support and clarification; otherwise identical to 1.1.

Version

    December 17, 2008 (version 1.2)

Number

    OpenGL Extension #222
    OpenGL ES Extension #52

Dependencies

    This extension is written against the OpenGL 1.2.1 Specification.
    It can also be used with OpenGL ES (see the section, "Dependencies on
    OpenGL ES," below).

Overview

    The goal of this extension is provide a finer granularity of
    synchronizing GL command completion than offered by standard OpenGL,
    which offers only two mechanisms for synchronization: Flush and Finish.
    Since Flush merely assures the user that the commands complete in a
    finite (though undetermined) amount of time, it is, thus, of only
    modest utility.  Finish, on the other hand, stalls CPU execution
    until all pending GL commands have completed.  This extension offers
    a middle ground - the ability to "finish" a subset of the command
    stream, and the ability to determine whether a given command has
    completed or not.

    This extension introduces the concept of a "fence" to the OpenGL
    command stream.  Once the fence is inserted into the command stream, it
    can be queried for a given condition - typically, its completion.
    Moreover, the application may also request a partial Finish -- that is,
    all commands prior to the fence will be forced to complete until control
    is returned to the calling process.  These new mechanisms allow for
    synchronization between the host CPU and the GPU, which may be accessing
    the same resources (typically memory).

    This extension is useful in conjunction with NV_vertex_array_range
    to determine when vertex information has been pulled from the
    vertex array range.  Once a fence has been tested TRUE or finished,
    all vertex indices issued before the fence must have been pulled.
    This ensures that the vertex data memory corresponding to the issued
    vertex indices can be safely modified (assuming no other outstanding
    vertex indices are issued subsequent to the fence).
    
Issues

    Do we need an IsFenceNV command?

        RESOLUTION:  Yes.  Not sure who would use this, but it's in there.
        Semantics currently follow the texture object definition --
        that is, calling IsFenceNV before SetFenceNV will return FALSE.

    Are the fences sharable between multiple contexts?

        RESOLUTION:  No.

        Potentially this could change with a subsequent extension.

    What other conditions will be supported?

        Only ALL_COMPLETED_NV will be supported initially.  Future extensions
        may wish to implement additional fence conditions.

    What is the relative performance of the calls?

        Execution of a SetFenceNV is not free, but will not trigger a
        Flush or Finish.

    Is the TestFenceNV call really necessary?  How often would this be used
    compared to the FinishFenceNV call (which also flushes to ensure this
    happens in finite time)?

        It is conceivable that a user may use TestFenceNV to decide
        which portion of memory should be used next without stalling
        the CPU.  An example of this would be a scenario where a single
        AGP buffer is used for both static (unchanged for multiple frames)
        and dynamic (changed every frame) data.  If the user has written
        dynamic data to all banks dedicated to dynamic data, and still
        has more dynamic objects to write, the user would first want to
        check if the first dynamic object has completed, before writing
        into the buffer.  If the object has not completed, instead of
        stalling the CPU with a FinishFenceNV call, it would possibly
        be better to start overwriting static objects instead.

    What should happen if TestFenceNV is called for a name before SetFenceNV
    is called?

        We generate an INVALID_OPERATION error, and return TRUE.
        This follows the semantics for texture object names before
        they are bound, in that they acquire their state upon binding.
        We will arbitrarily return TRUE for consistency.

    What should happen if FinishFenceNV is called for a name before
    SetFenceNV is called?

        RESOLUTION:  Generate an INVALID_OPERATION error because the
        fence id does not exist yet.  SetFenceNV must be called to create
        a fence.

    Do we need a mechanism to query which condition a given fence was
    set with?

        RESOLUTION:  Yes, use glGetFenceivNV with FENCE_CONDITION_NV.

    Should we allow these commands to be compiled within display list?
    Which ones?  How about within Begin/End pairs?

        RESOLUTION:  DeleteFencesNV, FinishFenceNV, GenFencesNV,
        TestFenceNV, and IsFenceNV are executed immediately while
        SetFenceNV is compiled.  Do not allow any of these commands
        within Begin/End pairs.

    Can fences be used as a form of performance monitoring?

        Yes, with some caveats.  By setting and testing or finishing
        fences, developers can measure the GPU latency for completing
        GL operations.  For example, developers might do the following:

          start = getCurrentTime();
          updateTextures();
          glSetFenceNV(TEXTURE_LOAD_FENCE, GL_ALL_COMPLETED_NV);
          drawBackground();
          glSetFenceNV(DRAW_BACKGROUND_FENCE, GL_ALL_COMPLETED_NV);
          drawCharacters();
          glSetFenceNV(DRAW_CHARACTERS_FENCE, GL_ALL_COMPLETED_NV);

          glFinishFenceNV(TEXTURE_LOAD_FENCE);
          textureLoadEnd = getCurrentTime();

          glFinishFenceNV(DRAW_BACKGROUND_FENCE);
          drawBackgroundEnd = getCurrentTime();

          glFinishFenceNV(DRAW_CHARACTERS_FENCE);
          drawCharactersEnd = getCurrentTime();

          printf("texture load time = %d\n", textureLoadEnd - start);
          printf("draw background time = %d\n", drawBackgroundEnd - textureLoadEnd);
          printf("draw characters time = %d\n", drawCharacters - drawBackgroundEnd);

        Note that there is a small amount of overhead associated with
        inserting each fence into the GL command stream.  Each fence
        causes the GL command stream to momentarily idle (idling the
        entire GPU pipeline).  The significance of this idling should
        be small if there are a small number of fences and large amount
        of intervening commands.

        If the time between two fences is zero or very near zero,
        it probably means that a GPU-CPU synchronization such as a
        glFinish probably occurred.  A glFinish is an explicit GPU-CPU
        synchronization, but sometimes implicit GPU-CPU synchronizations
        are performed by the driver.

    What happens if you set the same fence object twice?

        The second SetFenceNV clobbers whatever status the fence object
        previously had by forcing the object's status to GL_TRUE.
        The completion of the first SetFenceNV's fence command placed
        in the command stream is ignored (its completion does NOT
        update the fence object's status).  The second SetFenceNV sets a
        new fence command in the GL command stream.  This second fence
        command will update the fence object's status (assuming it is
        not ignored by a subsequent SetFenceNV to the same fence object).

    What happens to a fence command that is still pending execution
    when its fence object is deleted?

        The fence command completion is ignored.

    What happens if you use an arbitrary number for the SetFenceNV() <fence>
    parameter instead of obtaining the name from GenFences()?

        This works fine (just as with texture objects).

New Procedures and Functions

    void GenFencesNV(sizei n, uint *fences);

    void DeleteFencesNV(sizei n, const uint *fences);

    void SetFenceNV(uint fence, enum condition);

    boolean TestFenceNV(uint fence);

    void FinishFenceNV(uint fence);

    boolean IsFenceNV(uint fence);

    void GetFenceivNV(uint fence, enum pname, int *params);

New Tokens

    Accepted by the <condition> parameter of SetFenceNV:

        ALL_COMPLETED_NV                   0x84F2

    Accepted by the <pname> parameter of GetFenceivNV:

        FENCE_STATUS_NV                    0x84F3
        FENCE_CONDITION_NV                 0x84F4

Additions to Chapter 5 of the OpenGL 1.2.1 Specification (Special Functions)

    Add to the end of Section 5.4 "Display Lists"

    "DeleteFencesNV, FinishFenceNV, GenFencesNV, GetFenceivNV,
    TestFenceNV, and IsFenceNV are not complied into display lists but
    are executed immediately."

    After the discussion of Flush and Finish (Section 5.5) add a
    description of the fence operations:

    "5.X  Fences

    The command 

       void SetFenceNV(uint fence, enum condition);

    creates a fence object named <fence> if one does not already exist
    and sets a fence command within the GL command stream.  If the named
    fence object already exists, a new fence command is set within the GL
    command stream (and any previous pending fence command corresponding
    to the fence object is ignored).  Whether or not a new fence object is
    created, SetFenceNV assigns the named fence object a status of FALSE
    and a condition as set by the condition argument.  The condition
    argument must be ALL_COMPLETED_NV.  Once the fence's condition is
    satisfied within the command stream, the corresponding fence object's
    state is changed to TRUE.  For a condition of ALL_COMPLETED_NV,
    this is completion of the fence command and all preceding commands.
    No other state is affected by execution of the fence command.  The name
    <fence> may be one returned by GenFencesNV() but that is not required.

    A fence's state can be queried by calling the command

      boolean TestFenceNV(uint fence);

    The command

      void FinishFenceNV(uint fence);

    forces all GL commands prior to the fence to satisfy the condition
    set within SetFenceNV, which, in this spec, is always completion.
    FinishFenceNV does not return until all effects from these commands
    on GL client and server state and the framebuffer are fully realized.

    The command

      void GenFencesNV(sizei n, uint *fences);

    returns n previously unused fence names in fences.  These names
    are marked as used, for the purposes of GenFencesNV only, but
    corresponding fence objects do not exist (have no state) until created
    with SetFenceNV().

    Fences are deleted by calling

      void DeleteFencesNV(sizei n, const uint *fences);

    fences contains n names of fences to be deleted.  After a fence is
    deleted, it has no state, and its name is again unused.  Unused names
    in fences are silently ignored.

    If the fence passed to TestFenceNV or FinishFenceNV is not the name of an
    existing fence, the error INVALID_OPERATION is generated.  In this case,
    TestFenceNV will return TRUE, for the sake of consistency.

    State must be maintained to indicate which fence integers are
    currently used or set.  In the initial state, no indices are in use.
    When a fence integer is set, the condition and status of the fence
    are also maintained.  The status is a boolean.  The condition is
    the value last set as the condition by SetFenceNV.

    Once the status of a fence has been finished (via FinishFenceNV)
    or tested and the returned status is TRUE (via either TestFenceNV
    or GetFenceivNV querying the FENCE_STATUS_NV), the status remains
    TRUE until the next SetFenceNV of the fence."

Additions to Chapter 6 of the OpenGL 1.2.1 Specification (State and State Requests)

    Insert new section after Section 6.1.10 "Minmax Query"

    "6.1.11 Fence Query

    The command

      boolean IsFenceNV(uint fence);

    return TRUE if <fence> is the name of an existing fence.  If <fence> is
    not the name of an existing fence, or if an error condition occurs,
    IsFenceNV returns FALSE.  A name returned by GenFencesNV, but not yet set
    via SetFenceNV, is not the name of an existing fence.

    The command

      void GetFenceivNV(uint fence, enum pname, int *params)

    obtains the indicated fence state for the specified fence in the array
    params.  pname must be either FENCE_STATUS_NV or FENCE_CONDITION_NV.
    The INVALID_OPERATION error is generated if the named fence does
    not exist."

Additions to the GLX Specification

    None

GLX Protocol

    Seven new GL commands are added.

    The following rendering command is sent to the sever as part of a
    glXRender request:

        SetFenceNV
            2           12              rendering command length
            2           4143            rendering command opcode
            4           CARD32          fence
            4           CARD32          condition

    The remaining five commands are non-rendering commands.  These
    commands are sent separately (i.e., not as part of a glXRender or
    glXRenderLarge request), using the glXVendorPrivateWithReply request:

        DeleteFencesNV
            1           CARD8           opcode (X assigned)
            1           17              GLX opcode (glXVendorPrivateWithReply)
            2           4+n             request length
            4           1276            vendor specific opcode
            4           GLX_CONTEXT_TAG context tag
            4           INT32           n
            n*4         LISTofCARD32    fences

        GenFencesNV
            1           CARD8           opcode (X assigned)
            1           17              GLX opcode (glXVendorPrivateWithReply)
            2           4               request length
            4           1277            vendor specific opcode
            4           GLX_CONTEXT_TAG context tag
            4           INT32           n
          =>
            1           1               reply
            1                           unused
            2           CARD16          sequence number
            4           n               reply length
            24                          unused
            n*4         LISTofCARD322   fences

        IsFenceNV
            1           CARD8           opcode (X assigned)
            1           17              GLX opcode (glXVendorPrivateWithReply)
            2           4               request length
            4           1278            vendor specific opcode
            4           GLX_CONTEXT_TAG context tag
            4           INT32           n
          =>
            1           1               reply
            1                           unused
            2           CARD16          sequence number
            4           0               reply length
            4           BOOL32          return value
            20                          unused

        TestFenceNV
            1           CARD8           opcode (X assigned)
            1           17              GLX opcode (glXVendorPrivateWithReply)
            2           4               request length
            4           1279            vendor specific opcode
            4           GLX_CONTEXT_TAG context tag
            4           INT32           fence
          =>
            1           1               reply
            1                           unused
            2           CARD16          sequence number
            4           0               reply length
            4           BOOL32          return value
            20                          unused

        GetFenceivNV
            1           CARD8           opcode (X assigned)
            1           17              GLX opcode (glXVendorPrivateWithReply)
            2           5               request length
            4           1280            vendor specific opcode
            4           GLX_CONTEXT_TAG context tag
            4           INT32           fence
            4           CARD32          pname
          =>
            1           1               reply
            1                           unused
            2           CARD16          sequence number
            4           m               reply length, m=(n==1?0:n)
            4                           unused
            4           CARD32          n

            if (n=1) this follows:

            4           INT32           params
            12                          unused

            otherwise this follows:

            16                          unused
            n*4         LISTofINT32     params

        FinishFenceNV
            1           CARD8           opcode (X assigned)
            1           17              GLX opcode (glXVendorPrivateWithReply)
            2           4               request length
            4           1312            vendor specific opcode
            4           GLX_CONTEXT_TAG context tag
            4           INT32           fence
          =>
            1           1               reply
            1                           unused
            2           CARD16          sequence number
            4           0               reply length
            24                          unused

Errors

    INVALID_VALUE is generated if GenFencesNV or DeleteFencesNV parameter <n>
    is negative.

    INVALID_OPERATION is generated if the fence used in TestFenceNV,
    FinishFenceNV or GetFenceivNV is not the name of an existing fence.

    INVALID_ENUM is generated if the condition used in SetFenceNV
    is not ALL_COMPLETED_NV.

    INVALID_OPERATION is generated if any of the commands defined in
    this extension is executed between the execution of Begin and the
    corresponding execution of End.

New State

Table 6.X.  Fence Objects.

Get value           Type  Get command   Initial value                 Description      Section  Attribute
------------------  ----  ------------  ----------------------------  ---------------  -------  ---------
FENCE_STATUS_NV     B     GetFenceivNV  determined by 1st SetFenceNV  Fence status     5.X      -
FENCE_CONDITION_NV  Z1    GetFenceivNV  determined by 1st SetFenceNV  Fence condition  5.X      -

New Implementation Dependent State

    None

Dependencies on OpenGL ES

    If implemented for OpenGL ES, NV_fence acts as described in this spec,
    except:

        * Ignore all references to display lists and immediate mode, including
          changes to section 5.4 "Display Lists".
        * Ignore all references to GLX and GLX protocol.

GeForce Implementation Details

    This section describes implementation-defined limits for GeForce:

        SetFenceNV calls are not free.  They should be used prudently,
        and a "good number" of commands should be sent between calls to
        SetFenceNV.  Each fence insertion will cause the GPU's command
        processing to go momentarily idle.  Testing or finishing a fence
        may require an one or more somewhat expensive uncached reads.

        Do not leave a fence untested or unfinished for an extremely large
        interval of intervening fences.  If more than approximately 2
        billion (specifically 2^31-1) intervening fences are inserted into
        the GL command stream before a fence is tested or finished, said
        fence may indicate an incorrect status.  Note that certain GL
        operations involving display lists, compiled vertex arrays, and
        textures may insert fences implicitly for internal driver use.

        In practice, this limitation is unlikely to be a practical
        limitation if fences are finished or tested within a few frames
        of their insertion into the GL command stream.

Revision History

    November 13, 2000 - GLX enumerant values assigned

    October 3, 2003 - Changed version to 1.1.  glFinishFenceNV should
    not be compiled into display lists but rather executed immediately
    when called during display list construction.  Version 1.0 allowed
    this though it should not have been allowed.  Changed GLX protocol
    so that FinishFenceNV is a non-render request with a reply now.
    Thanks to Bob Beretta for noticing this issue.

    Also fix a typo in the GLX protocol specification for IsFenceNV so
    the reply is 32 (not 33) bytes.

    December 17. 2008 - Add "Dependencies on OpenGL ES" section.  Clarify
    generation of fence name vs creation of the fence itself.
