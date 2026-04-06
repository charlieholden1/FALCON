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

#ifndef PCOUNT3D_HWRES_H
#define PCOUNT3D_HWRES_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <ti/drivers/edma/edma.h>
#include <ti/common/sys_common.h>

#define EDMA_INSTANCE_0 0
#define EDMA_INSTANCE_1 1

/*******************************************************************************
 * Resources for Object Detection DPC, currently the only DPC and hwa/edma
 * resource user in the demo.
 *******************************************************************************/
/* EDMA instance used for objdet that is managed by R4F - this is for HWA 1D. */
#define DPC_OBJDET_R4F_EDMA_INSTANCE EDMA_INSTANCE_0

/* Shadow base for above instance */
#define DPC_OBJDET_R4F_EDMA_SHADOW_BASE EDMA_NUM_DMA_CHANNELS

/* EDMA instance used for objdet that is managed by DSP */
#define DPC_OBJDET_DSP_EDMA_INSTANCE EDMA_INSTANCE_1

/* Shadow base for above instance */
#define DPC_OBJDET_DSP_EDMA_SHADOW_BASE EDMA_NUM_DMA_CHANNELS

#define DPC_OBJDET_HWA_WINDOW_RAM_OFFSET 0
#define DPC_OBJDET_PARAMSET_START_IDX    0

/* Range DPU */
#define DPC_OBJDET_DPU_RANGEPROC_PARAMSET_START_IDX DPC_OBJDET_PARAMSET_START_IDX

#define DPC_OBJDET_DPU_RANGEPROC_EDMA_INST_ID         DPC_OBJDET_R4F_EDMA_INSTANCE
#define DPC_OBJDET_DPU_RANGEPROC_EDMAIN_CH            EDMA_TPCC0_REQ_DFE_CHIRP_AVAIL
#define DPC_OBJDET_DPU_RANGEPROC_EDMAIN_SHADOW        (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 0)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAIN_EVENT_QUE     0
#define DPC_OBJDET_DPU_RANGEPROC_EDMAIN_SIG_CH        EDMA_TPCC0_REQ_FREE_0
#define DPC_OBJDET_DPU_RANGEPROC_EDMAIN_SIG_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 1)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAIN_SIG_EVENT_QUE 0

/* FMT2 EDMA resources */
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PING_CH        EDMA_TPCC0_REQ_HWACC_0
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PING_SHADOW_0  (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 2)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PING_SHADOW_1  (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 3)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PING_SHADOW_2  (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 4)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PING_EVENT_QUE 0

#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PINGDATA_0_CH        EDMA_TPCC0_REQ_FREE_1
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PINGDATA_0_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 5)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PINGDATA_0_EVENT_QUE 0
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PINGDATA_1_CH        EDMA_TPCC0_REQ_FREE_2
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PINGDATA_1_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 6)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PINGDATA_1_EVENT_QUE 0
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PINGDATA_2_CH        EDMA_TPCC0_REQ_FREE_3
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PINGDATA_2_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 7)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PINGDATA_2_EVENT_QUE 0

#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONG_CH        EDMA_TPCC0_REQ_HWACC_1
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONG_SHADOW_0  (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 8)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONG_SHADOW_1  (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 9)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONG_SHADOW_2  (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 10)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONG_EVENT_QUE 0

#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONGDATA_0_CH        EDMA_TPCC0_REQ_FREE_4
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONGDATA_0_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 11)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONGDATA_0_EVENT_QUE 0
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONGDATA_1_CH        EDMA_TPCC0_REQ_FREE_5
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONGDATA_1_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 12)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONGDATA_1_EVENT_QUE 0
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONGDATA_2_CH        EDMA_TPCC0_REQ_FREE_6
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONGDATA_2_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 13)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT2_PONGDATA_2_EVENT_QUE 0

/* FMT1 EDMA resources */
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PING_CH        EDMA_TPCC0_REQ_HWACC_0
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PING_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 2)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PING_EVENT_QUE 0

#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PONG_CH        EDMA_TPCC0_REQ_HWACC_1
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PONG_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 3)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_FMT1_PONG_EVENT_QUE 0

#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_SIG_CH        EDMA_TPCC0_REQ_FREE_7
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_SIG_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 14)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_SIG_EVENT_QUE 0

#define UART_DMA_TX_CHANNEL 1
#define UART_DMA_RX_CHANNEL 2

/* Minor Mode EDMA resources */
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_MINOR_CH        EDMA_TPCC0_REQ_FREE_8
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_MINOR_SHADOW_0  (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 15)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_MINOR_SHADOW_1  (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 16)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_MINOR_SHADOW_2  (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 17)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_MINOR_EVENT_QUE 0

#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_MINORDATA_CH        EDMA_TPCC0_REQ_FREE_9
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_MINORDATA_SHADOW    (DPC_OBJDET_R4F_EDMA_SHADOW_BASE + 18)
#define DPC_OBJDET_DPU_RANGEPROC_EDMAOUT_MINORDATA_EVENT_QUE 0

/* DSP copy table EDMA resources */
#define DPC_OBJDET_DPU_CPTB_PROC_EDMA_INST_ID        DPC_OBJDET_DSP_EDMA_INSTANCE
#define DPC_OBJDET_DPU_CPTB_PROC_EDMA_PING_CH        EDMA_TPCC1_REQ_FREE_13
#define DPC_OBJDET_DPU_CPTB_PROC_EDMA_PING_SHADOW    (DPC_OBJDET_DSP_EDMA_SHADOW_BASE + 12)
#define DPC_OBJDET_DPU_CPTB_PROC_EDMA_PING_EVENT_QUE 0

#define DPC_OBJDET_DPU_CPTB_PROC_EDMA_PONG_CH        EDMA_TPCC1_REQ_FREE_14
#define DPC_OBJDET_DPU_CPTB_PROC_EDMA_PONG_SHADOW    (DPC_OBJDET_DSP_EDMA_SHADOW_BASE + 13)
#define DPC_OBJDET_DPU_CPTB_PROC_EDMA_PONG_EVENT_QUE 0

#ifdef __cplusplus
}
#endif

#endif /* PCOUNT3D_HWRES_H */
