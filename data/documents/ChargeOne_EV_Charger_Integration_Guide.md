---
document_id: VE-EVC-IG-002
title: ChargeOne EV Charger Integration Guide
version: 2.0
status: approved
lifecycle_status: current
effective_date: 2026-05-20
owner: Product Engineering
region: DE
product_family: ChargeOne
document_type: integration_guide
---

# ChargeOne EV Charger Integration Guide

## 1. Purpose
This document provides integration guidance for the ChargeOne 11 residential EV charger in VoltEdge residential energy systems.

## 2. Supported Product
This guide applies to ChargeOne 11.

## 3. Operating Modes
### 3.1 Standard Charging
ChargeOne 11 can provide standard EV charging without EnergyHub when installed according to approved installation instructions. In this mode, charging does not dynamically respond to PV generation, battery state of charge, or household consumption.

### 3.2 Coordinated Energy Management
Functions such as PV-surplus charging and dynamic charging optimization require integration with EnergyHub. EnergyHub must have access to current household consumption and PV-generation data.

## 4. PV-Surplus Charging
PV-surplus charging is available only when ChargeOne 11 is integrated with EnergyHub and the required metering data is available. A HomeCell battery is not required. If a HomeCell battery is installed, battery and EV charging are coordinated according to the active EnergyHub operating strategy.

## 5. Retrofit Installations
When ChargeOne 11 is added to an existing VoltEdge installation, the installer must:
1. verify the existing system configuration;
2. verify required metering data;
3. confirm connectivity between ChargeOne 11 and EnergyHub where coordinated functions are required;
4. complete charger commissioning;
5. verify communication status and operating mode before handover.

Existing third-party metering or control equipment may affect coordinated functions. Unsupported configurations require technical review.

## 6. Communication Requirements
Loss of communication with EnergyHub does not prevent basic charging, but coordinated functions such as PV-surplus charging may become unavailable until communication is restored.

## 7. Commissioning
For coordinated operation:
1. Commission the inverter and, where applicable, HomeCell.
2. Verify EnergyHub receives valid household and PV-generation data.
3. Commission ChargeOne 11.
4. Add ChargeOne 11 to EnergyHub.
5. Verify charger communication.
6. Activate the required charging mode.
7. Confirm correct operation.

The system must not be handed over with unresolved communication or commissioning errors.

## 8. Limitations
ChargeOne 11 does not independently determine whole-home PV surplus. PV-surplus charging must not be represented as available unless EnergyHub integration and metering prerequisites are fulfilled.

## 9. Warranty and Compliance
Warranty conditions are governed by the applicable regional warranty policy. This guide does not override warranty terms.

## 10. Related Documents
- EnergyHub System Integration Guide
- Approved Product Compatibility Data
- HomeCell Battery Installation and Commissioning Guide
- VoltEdge Residential Energy Warranty Policy – Germany
