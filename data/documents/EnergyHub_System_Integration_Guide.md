---
document_id: VE-EH-SIG-003
title: EnergyHub System Integration Guide
version: 3.0
status: approved
lifecycle_status: current
effective_date: 2026-06-10
owner: System Engineering
region: DE
product_family: EnergyHub
document_type: system_integration_guide
---

# EnergyHub System Integration Guide

## 1. Purpose
This document describes the role of EnergyHub in VoltEdge residential energy systems and the prerequisites for coordinated operation across supported devices.

## 2. System Role
EnergyHub is the coordination layer for supported energy-management functions and may use data from:
- the VoltEdge hybrid inverter;
- the household energy meter;
- HomeCell battery systems;
- ChargeOne 11 EV chargers.

EnergyHub does not replace device-level safety, protection, or product compatibility requirements.

## 3. Required Metering Data
Coordinated functions require reliable household grid import/export data. PV-surplus charging additionally requires current PV-generation information. If required data is unavailable, EnergyHub must not activate dependent functions.

## 4. Supported Energy-Management Functions
When prerequisites are met, EnergyHub supports:
- monitoring of household energy flows;
- coordination of HomeCell charging and discharging;
- PV-surplus charging with ChargeOne 11;
- prioritization of household consumption before optional flexible loads;
- visualization of system status.

## 5. Integration with ChargeOne 11
Before coordinated charging is activated, the installer must verify:
1. ChargeOne 11 is commissioned and reachable;
2. EnergyHub receives valid household meter data;
3. EnergyHub receives valid PV-generation data;
4. the charger is assigned to the correct residential energy system;
5. there are no unresolved communication errors.

Basic EV charging can remain available independently of coordinated EnergyHub functions.

## 6. Integration with HomeCell
When HomeCell is present, EnergyHub uses battery status information when coordinating flexible energy consumption.

EnergyHub does not determine whether a specific inverter and battery combination is approved. Product compatibility and minimum firmware requirements must be verified using current approved product compatibility data.

## 7. Example Operating Logic
In a system with PV generation, HomeCell, and ChargeOne 11:
1. current household demand is measured;
2. available PV generation is determined;
3. the configured operating strategy determines how available energy is allocated between battery charging and EV charging;
4. grid import or export adjusts according to system conditions and configured limits.

## 8. Communication and Connectivity
Temporary loss of communication may cause coordinated functions to pause or fall back to device-level behavior. Persistent failures must be resolved before handover.

## 9. Retrofit and Third-Party Equipment
For retrofit installations, the installer must verify that existing metering and control architecture can provide the data required by EnergyHub. Unsupported control architectures or unvalidated data sources require technical review.

## 10. System Boundaries
EnergyHub does not:
- override product-level compatibility rules;
- replace regional installation requirements;
- establish warranty eligibility;
- validate unsupported third-party configurations.

## 11. Related Documents
- ChargeOne EV Charger Integration Guide
- HomeCell Battery Installation and Commissioning Guide
- Approved Product Compatibility Data
- VoltEdge Residential Energy Warranty Policy – Germany
