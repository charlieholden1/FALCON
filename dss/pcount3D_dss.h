/*
 * Copyright (C) 2024 Texas Instruments Incorporated
 *
 * All rights reserved not granted herein.
 * Limited License.
 *
 * Texas Instruments Incorporated grants a world-wide, royalty-free,
 * non-exclusive license under copyrights and patents it now or hereafter
 * owns or controls to make, have made, use, import, offer to sell and sell ("Utilize")
 * this software subject to the terms herein.  With respect to the foregoing patent
 * license, such license is granted  solely to the extent that any such patent is necessary
 * to Utilize the software alone.  The patent license shall not apply to any combinations which
 * include this software, other than combinations with devices manufactured by or for TI ("TI Devices").
 * No hardware patent is licensed hereunder.
 *
 * Redistributions must preserve existing copyright notices and reproduce this license (including the
 * above copyright notice and the disclaimer and (if applicable) source code license limitations below)
 * in the documentation and/or other materials provided with the distribution
 *
 * Redistribution and use in binary form, without modification, are permitted provided that the following
 * conditions are met:
 *
 *	* No reverse engineering, decompilation, or disassembly of this software is permitted with respect to any
 *     software provided in binary form.
 *	* any redistribution and use are licensed by TI for use only with TI Devices.
 *	* Nothing shall obligate TI to provide you with source code for the software licensed and provided to you in object code.
 *
 * If software source code is provided to you, modification and redistribution of the source code are permitted
 * provided that the following conditions are met:
 *
 *   * any redistribution and use of the source code, including any resulting derivative works, are licensed by
 *     TI for use only with TI Devices.
 *   * any redistribution and use of any object code compiled from the source code and any resulting derivative
 *     works, are licensed by TI for use only with TI Devices.
 *
 * Neither the name of Texas Instruments Incorporated nor the names of its suppliers may be used to endorse or
 * promote products derived from this software without specific prior written permission.
 *
 * DISCLAIMER.
 *
 * THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PCOUNT3D_DSS_H
#define PCOUNT3D_DSS_H

#include <ti/drivers/edma/edma.h>
#include <cpy_tbl.h>

#include <ti/sysbios/knl/Task.h>

#include <ti/common/mmwave_error.h>
#include <ti/drivers/soc/soc.h>
#include <ti/drivers/soc/soc.h>
#include <ti/drivers/mailbox/mailbox.h>
#include <ti/drivers/hwa/hwa.h>
#include <ti/drivers/osal/DebugP.h>
#include <ti/demo/xwr68xx/mmw/include/mmw_output.h>

#include "pcount3D_config.h"

#include <dpu/capon3d/radarProcess.h>
#include <dpu/capon3d/modules/utilities/cycle_measure.h>

/* This is used to resolve RL_MAX_SUBFRAMES, TODO: wired */
#include <ti/control/mmwavelink/mmwavelink.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct Pcount3DDemo_DataPathObj_t
    {
        /*! @brief dpm Handle */
        DPM_Handle radarProcDpmHandle;

        /*! @brief   Handle of the EDMA driver. */
        EDMA_Handle edmaHandle;

        /*! @brief   EDMA error Information when there are errors like missing events */
        EDMA_errorInfo_t EDMA_errorInfo;

        /*! @brief EDMA transfer controller error information. */
        EDMA_transferControllerErrorInfo_t EDMA_transferControllerErrorInfo;

        /*! @brief          Processing Stats */
        MmwDemo_output_message_stats subFrameStats[RL_MAX_SUBFRAMES];
    } Pcount3DDemo_DataPathObj;

    /**
     * @brief
     *  Millimeter Wave Demo MCB
     *
     * @details
     *  The structure is used to hold all the relevant information for the
     *  Millimeter Wave demo
     */
    typedef struct Pcount3DDemo_DSS_MCB_t
    {
        /*! * @brief   Handle to the SOC Module */
        SOC_Handle socHandle;

        /*! @brief     DPM Handle */
        Task_Handle radarProcDpmTaskHandle;

        /*! @brief     init Task Handle */
        Task_Handle initTaskHandle;

        /*! @brief     Data Path object */
        Pcount3DDemo_DataPathObj dataPathObj;

        /*! @brief   Counter which tracks the number of dpm stop events received
                     The event is triggered by DPM_Report_DPC_STOPPED from DPM */
        uint32_t dpmStopEvents;

        /*! @brief   Counter which tracks the number of dpm start events received
                     The event is triggered by DPM_Report_DPC_STARTED from DPM */
        uint32_t dpmStartEvents;
    } Pcount3DDemo_DSS_MCB;


    /**************************************************************************
     *************************** Extern Definitions ***************************
     **************************************************************************/
    extern void Pcount3DDemo_dataPathInit(Pcount3DDemo_DataPathObj *obj);
    extern void Pcount3DDemo_dataPathOpen(Pcount3DDemo_DataPathObj *obj);
    extern void Pcount3DDemo_dataPathClose(Pcount3DDemo_DataPathObj *obj);

    /* Sensor Management Module Exported API */
    extern void _Pcount3DDemo_debugAssert(int32_t expression, const char *file, int32_t line);
#define Pcount3DDemo_debugAssert(expression) \
    { \
        DebugP_assert(expression); \
    }

#ifdef __cplusplus
}
#endif

#endif /* PCOUNT3D_DSS_H */
