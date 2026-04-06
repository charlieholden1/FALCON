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

#ifndef MMW_MSS_H
#define MMW_MSS_H

#include <ti/sysbios/knl/Semaphore.h>
#include <ti/sysbios/knl/Task.h>

#include <ti/common/mmwave_error.h>
#include <ti/drivers/osal/DebugP.h>
#include <ti/drivers/soc/soc.h>
#include <ti/drivers/uart/UART.h>
#include <ti/drivers/gpio/gpio.h>
#include <ti/drivers/mailbox/mailbox.h>

#include "common/mmwdemo_adcconfig.h"

#include <dpc/objdetrangehwa/objdetrangehwa.h>
#include "pcount3D_config.h"
#include "mmwdemo_tlv.h"


#ifdef __cplusplus
extern "C"
{
#endif

/*! @brief For advanced frame config, below define means the configuration given is
 * global at frame level and therefore it is broadcast to all sub-frames.
 */
#define PCOUNT3DDEMO_SUBFRAME_NUM_FRAME_LEVEL_CONFIG (-1)

/**
 * @defgroup configStoreOffsets     Offsets for storing CLI configuration
 * @brief    Offsets of config fields within the parent structures, note these offsets will be
 *           unique and hence can be used to differentiate the commands for processing purposes.
 * @{
 */
#define PCOUNT3DDEMO_ADCBUFCFG_OFFSET (offsetof(Pcount3DDemo_SubFrameCfg, adcBufCfg))

#define PCOUNT3DDEMO_SUBFRAME_DSPDYNCFG_OFFSET (offsetof(Pcount3DDemo_SubFrameCfg, objDetDynCfg) + \
                                                offsetof(Pcount3DDemo_DPC_ObjDet_DynCfg, dspDynCfg))

#define PCOUNT3DDEMO_SUBFRAME_R4FDYNCFG_OFFSET (offsetof(Pcount3DDemo_SubFrameCfg, objDetDynCfg) + \
                                                offsetof(Pcount3DDemo_DPC_ObjDet_DynCfg, r4fDynCfg))

#define PCOUNT3DDEMO_CALIBDCRANGESIG_OFFSET (PCOUNT3DDEMO_SUBFRAME_R4FDYNCFG_OFFSET + \
                                             offsetof(DPC_ObjectDetectionRangeHWA_DynCfg, calibDcRangeSigCfg))

#define PCOUNT3DDEMO_CAPONCHAINCFG_OFFSET (PCOUNT3DDEMO_SUBFRAME_DSPDYNCFG_OFFSET + \
                                           offsetof(DPC_ObjectDetection_DynCfg, caponChainCfg))

#define PCOUNT3DDEMO_DYNRACFARCFG_OFFSET (PCOUNT3DDEMO_CAPONCHAINCFG_OFFSET + \
                                          offsetof(caponChainCfg, dynamicCfarConfig))

#define PCOUNT3DDEMO_STATICRACFARCFG_OFFSET (PCOUNT3DDEMO_CAPONCHAINCFG_OFFSET + \
                                             offsetof(caponChainCfg, staticCfarConfig))

#define PCOUNT3DDEMO_DOACAPONCFG_OFFSET (PCOUNT3DDEMO_CAPONCHAINCFG_OFFSET + \
                                         offsetof(caponChainCfg, doaConfig))

#define PCOUNT3DDEMO_DOACAPONRACFG_OFFSET (PCOUNT3DDEMO_DOACAPONCFG_OFFSET + \
                                           offsetof(doaConfig, rangeAngleCfg))

#define PCOUNT3DDEMO_DOA2DESTCFG_OFFSET (PCOUNT3DDEMO_DOACAPONCFG_OFFSET + \
                                         offsetof(doaConfig, angle2DEst))

#define PCOUNT3DDEMO_DOAFOVCFG_OFFSET (PCOUNT3DDEMO_DOACAPONCFG_OFFSET + \
                                       offsetof(doaConfig, fovCfg))

#define PCOUNT3DDEMO_STATICANGESTCFG_OFFSET (PCOUNT3DDEMO_DOACAPONCFG_OFFSET + \
                                             offsetof(doaConfig, staticEstCfg))

#define PCOUNT3DDEMO_DOPCFARCFG_OFFSET (PCOUNT3DDEMO_DOACAPONCFG_OFFSET + \
                                        offsetof(doaConfig, dopCfarCfg))


    typedef struct MmwDemo_output_message_compressedPointCloud_uart_t
    {
        MmwDemo_output_message_tl                   header;
        MmwDemo_output_message_compressedPoint_unit pointUint;
        MmwDemo_output_message_compressedPoint      point[MAX_RESOLVED_OBJECTS_PER_FRAME];
    } MmwDemo_output_message_compressedPointCloud_uart;


    /** @}*/ /* configStoreOffsets */

    /**
     * @brief
     *  3D people counting Demo Sensor State
     *
     * @details
     *  The enumeration is used to define the sensor states used in 3D people counting Demo
     */
    typedef enum Pcount3DDemo_SensorState_e
    {
        /*!  @brief Inital state after sensor is initialized.
         */
        Pcount3DDemo_SensorState_INIT = 0,

        /*!  @brief Indicates sensor is started */
        Pcount3DDemo_SensorState_STARTED,

        /*!  @brief  State after sensor has completely stopped */
        Pcount3DDemo_SensorState_STOPPED
    } Pcount3DDemo_SensorState;

    /**
     * @brief
     *  3D people counting Demo statistics
     *
     * @details
     *  The structure is used to hold the statistics information for the
     *  3D people counting Demo
     */
    typedef struct Pcount3DDemo_MSS_Stats_t
    {
        /*! @brief   Counter which tracks the number of frame trigger events from BSS */
        uint64_t frameTriggerReady;

        /*! @brief   Counter which tracks the number of failed calibration reports
         *           The event is triggered by an asynchronous event from the BSS */
        uint32_t failedTimingReports;

        /*! @brief   Counter which tracks the number of calibration reports received
         *           The event is triggered by an asynchronous event from the BSS */
        uint32_t calibrationReports;

        /*! @brief   Counter which tracks the number of sensor stop events received
         *           The event is triggered by an asynchronous event from the BSS */
        uint32_t sensorStopped;

        /*! @brief   Counter which tracks the number of dpm stop events received
         *           The event is triggered by DPM_Report_DPC_STOPPED from DPM */
        uint32_t dpmStopEvents;

        /*! @brief   Counter which tracks the number of dpm start events received
         *           The event is triggered by DPM_Report_DPC_STARTED from DPM */
        uint32_t dpmStartEvents;

    } Pcount3DDemo_MSS_Stats;

    /**
     * @brief
     *  The structure specifies the BPM configuration.
     *
     * @details
     *  The BPM is supported only for two azimuth transmit antennas.
     *  say A and B. In the even time slots (0,2,..), both transmit antennas should be configured
     *  to transmit with positive phase i.e
     *  (A,B) = (+, +)
    .*  In the odd time slots (1,3..), the transmit antennas should be configured to transmit with phase
     *  (A,B) = (+, -)
     *
     *  The BPM decoding will produce the virtual antenna array in the order A,B [not B,A] which will
     *  be used for AoA processing. So user must make sure that the A,B mapping to the physical
     *  transmit antennas corresponds to the intended virtual antenna order. On the 6843 EVM, this
     *  means A = Tx1 and B = Tx3
     *
     *  Please note: mmwave link use TX antenna index starting from 0.
     */
    typedef struct Pcount3DDemo_BpmCfg_t
    {
        /**
         * @brief   Enabled/disabled flag
         */
        bool isEnabled;

        /**
         * @brief
         *          If BPM is enabled, this is the chirp index for the first BPM chirp.
         *          It will have phase 0 on both Azimuth TX antennas (+ +).
         *
         *          If BPM is disabled, a BPM disable command (set phase to zero) will
         *          be issued for the chirps in the range [chirp0Idx..chirp1Idx].
         */
        uint16_t chirp0Idx;

        /**
         * @brief
         *          If BPM is enabled, this is the chirp index for the second BPM chirp.
         *          It will have phase 0 on the first Azimuth TX antenna and phase 180
         *          on second Azimuth TX antenna (+ -).
         *
         *          If BPM is disabled, a BPM disable command (set phase to zero) will
         *          be issued for the chirps in the range [chirp0Idx..chirp1Idx].
         */
        uint16_t chirp1Idx;
    } Pcount3DDemo_BpmCfg;

    /**
     * @brief
     *  3D people counting Demo Data Path Information.
     *
     * @details
     *  The structure is used to hold all the relevant information for
     *  the data path.
     */
    typedef struct Pcount3DDemo_SubFrameCfg_t
    {
        /*! @brief ADC buffer configuration storage */
        MmwDemo_ADCBufCfg adcBufCfg;

        /*! @brief Flag indicating if @ref adcBufCfg is pending processing. */
        uint8_t isAdcBufCfgPending : 1;

        /*! @brief BPM configuration storage */
        Pcount3DDemo_BpmCfg bpmCfg;

        /*! @brief Dynamic configuration storage for object detection DPC */
        Pcount3DDemo_DPC_ObjDet_DynCfg objDetDynCfg;

        /*! @brief  ADCBUF will generate chirp interrupt event every this many chirps - chirpthreshold */
        uint8_t numChirpsPerChirpEvent;

        /*! @brief  Number of bytes per RX channel, it is aligned to 16 bytes as required by ADCBuf driver  */
        uint32_t adcBufChanDataSize;

        /*! @brief  Number of ADC samples */
        uint16_t numAdcSamples;

        /*! @brief  Number of chirps per sub-frame */
        uint16_t numChirpsPerSubFrame;

        /*! @brief  Number of virtual antennas */
        uint8_t numVirtualAntennas;
    } Pcount3DDemo_SubFrameCfg;

    /*!
     * @brief
     * Structure holds message stats information from data path.
     *
     * @details
     *  The structure holds stats information. This is a payload of the TLV message item
     *  that holds stats information.
     */
    typedef struct Pcount3DDemo_SubFrameStats_t
    {
        /*! @brief   Frame processing stats */
        MmwDemo_output_message_stats outputStats;

        /*! @brief   Dynamic CLI configuration time in usec */
        uint32_t pendingConfigProcTime;

        /*! @brief   SubFrame Preparation time on MSS in usec */
        uint32_t subFramePreparationTime;
    } Pcount3DDemo_SubFrameStats;

    /**
     * @brief Task handles storage structure
     */
    typedef struct Pcount3DDemo_TaskHandles_t
    {
        /*! @brief   MMWAVE Control Task Handle */
        Task_Handle mmwaveCtrl;

        /*! @brief   ObjectDetection DPC related dpmTask */
        Task_Handle objDetDpmTask;

        /*! @brief   Demo init task */
        Task_Handle initTask;
    } Pcount3DDemo_taskHandles;

    typedef struct Pcount3DDemo_DataPathObj_t
    {
        /*! @brief Handle to hardware accelerator driver. */
        HWA_Handle hwaHandle;

        /*! @brief   Handle of the EDMA driver. */
        EDMA_Handle edmaHandle;

        /*! @brief   Radar cube memory information from range DPC */
        DPC_ObjectDetectionRangeHWA_preStartCfg_radarCubeMem radarCubeMem;

        /*! @brief   Memory usage after the preStartCfg range DPC is applied */
        DPC_ObjectDetectionRangeHWA_preStartCfg_memUsage memUsage;

        /*! @brief   EDMA error Information when there are errors like missing events */
        EDMA_errorInfo_t EDMA_errorInfo;

        /*! @brief EDMA transfer controller error information. */
        EDMA_transferControllerErrorInfo_t EDMA_transferControllerErrorInfo;

    } Pcount3DDemo_DataPathObj;

    /**
     * @brief
     *  3D people counting Demo  MCB
     *
     * @details
     *  The structure is used to hold all the relevant information for the
     *  3D people counting Demo
     */
    typedef struct Pcount3DDemo_MSS_MCB_t
    {
        /*! @brief      Configuration which is used to execute the demo */
        Pcount3DDemo_Cfg cfg;

        /*! * @brief    Handle to the SOC Module */
        SOC_Handle socHandle;

        /*! @brief      UART Logging Handle */
        UART_Handle loggingUartHandle;

        /*! @brief      UART Command Rx/Tx Handle */
        UART_Handle commandUartHandle;

        /*! @brief      This is the mmWave control handle which is used
         * to configure the BSS. */
        MMWave_Handle ctrlHandle;

        /*! @brief      ADCBuf driver handle */
        ADCBuf_Handle adcBufHandle;

        /*! @brief      DSP chain DPM Handle */
        DPM_Handle objDetDpmHandle;

        /*! @brief      Object Detection DPC common configuration */
        Pcount3DDemo_DPC_ObjDet_CommonCfg objDetCommonCfg;

        /*! @brief      Data path object */
        Pcount3DDemo_DataPathObj dataPathObj;

        /*! @brief   Tracker DPU Static Configuration */
        DPC_ObjectDetection_TrackerConfig trackerCfg;

        /*! @brief      Object Detection DPC subFrame configuration */
        Pcount3DDemo_SubFrameCfg subFrameCfg[RL_MAX_SUBFRAMES];

        /*! @brief      sub-frame stats */
        Pcount3DDemo_SubFrameStats subFrameStats[RL_MAX_SUBFRAMES];

        /*! @brief      Demo Stats */
        Pcount3DDemo_MSS_Stats stats;

        MmwDemo_output_message_compressedPointCloud_uart pointCloudToUart;
        DPIF_DetMatrix                                   heatMapOutFromDSP;
        DPIF_PointCloudSpherical                        *pointCloudFromDSP;
        DPIF_PointCloudSideInfo                         *pointCloudSideInfoFromDSP;
        DPC_ObjectDetection_Stats                       *frameStatsFromDSP;
        uint32_t                                         currSubFrameIdx;

        trackerProc_TargetDescrHandle trackerOutput;
        uint8_t                       numTargets;
        uint16_t                      numIndices;
        bool                          presenceDetEnabled;
        uint32_t                      presenceInd;
        uint16_t                      numDetectedPoints;
        uint32_t                      trackerProcessingTimeInUsec;
        uint32_t                      uartProcessingTimeInUsec;


        /*! @brief      Task handle storage */
        Pcount3DDemo_taskHandles taskHandles;

        /*! @brief   RF frequency scale factor, = 2.7 for 60GHz device, = 3.6 for 76GHz device */
        double rfFreqScaleFactor;

        /*! @brief   Semaphore handle to signal DPM started from DPM report function */
        Semaphore_Handle DPMstartSemHandle;

        /*! @brief   Semaphore handle to signal DPM stopped from DPM report function. */
        Semaphore_Handle DPMstopSemHandle;

        /*! @brief   Semaphore handle to signal DPM ioctl from DPM report function. */
        Semaphore_Handle DPMioctlSemHandle;

        /*! @brief   Semaphore handle to run UART DMA task. */
        Semaphore_Handle uartTxSemHandle;

        /*! @brief   Semaphore handle to trigger tracker DPU. */
        Semaphore_Handle trackerDPUSemHandle;

        /*! @brief    Sensor state */
        Pcount3DDemo_SensorState sensorState;

        /*! @brief   Tracks the number of sensor start */
        uint32_t sensorStartCount;

        /*! @brief   Tracks the number of sensor sop */
        uint32_t sensorStopCount;

    } Pcount3DDemo_MSS_MCB;

    /**************************************************************************
     *************************** Extern Definitions ***************************
     **************************************************************************/

    /* Functions to handle the actions need to move the sensor state */
    extern int32_t Pcount3DDemo_openSensor(bool isFirstTimeOpen);
    extern int32_t Pcount3DDemo_configSensor(void);
    extern int32_t Pcount3DDemo_startSensor(void);
    extern void    Pcount3DDemo_stopSensor(void);

    /* functions to manage the dynamic configuration */
    extern uint8_t Pcount3DDemo_isAllCfgInPendingState(void);
    extern uint8_t Pcount3DDemo_isAllCfgInNonPendingState(void);
    extern void    Pcount3DDemo_resetStaticCfgPendingState(void);
    extern void    Pcount3DDemo_CfgUpdate(void *srcPtr, uint32_t offset, uint32_t size, int8_t subFrameNum);

    extern void Pcount3DDemo_CLIInit(uint8_t taskPriority);

    /* Debug Functions */
    extern void _Pcount3DDemo_debugAssert(int32_t expression, const char *file, int32_t line);
#define Pcount3DDemo_debugAssert(expression) \
    { \
        _Pcount3DDemo_debugAssert(expression, \
                                  __FILE__, \
                                  __LINE__); \
        DebugP_assert(expression); \
    }

#ifdef __cplusplus
}
#endif

#endif /* MMW_MSS_H */
