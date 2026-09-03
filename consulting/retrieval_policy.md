# Retrieval Policy

## 1. Purpose

This policy defines how the GenAI Sales Enablement Assistant identifies, selects, validates, and uses enterprise knowledge sources when answering user questions.

The policy does not redefine source ownership, source authority, or source-specific usage modes. Those are maintained in:

- `consulting/source_of_truth_mapping.csv`
- `consulting/knowledge_source_registry.csv`

This document defines how the system must apply those configurations during retrieval and answer generation.

## 2. Query and Knowledge-Domain Identification

For each user query, the system must identify the relevant knowledge domain or domains.

A query may involve more than one domain. For example, a retrofit question may simultaneously involve:

- product compatibility;
- firmware requirements;
- installation prerequisites;
- system integration;
- warranty conditions.

When multiple domains are identified, each domain must be resolved independently before the results are combined.

## 3. Authority Resolution

For each identified knowledge domain, the system must resolve the primary authoritative source using:

`consulting/source_of_truth_mapping.csv`

Supporting sources may be used only where they are explicitly listed or where their use is consistent with the resolution rule for that domain.

Supporting sources must not override the primary authoritative source.

If two authoritative sources provide conflicting information and no resolution rule is available, the system must not resolve the conflict by inference. The case must be treated as unresolved and escalated.

## 4. Source Eligibility

After resolving the relevant source, the system must retrieve the source configuration from:

`consulting/knowledge_source_registry.csv`

A source is eligible for use only if all configured requirements are met.

At minimum:

- the source approval status must satisfy `required_approval_status`;
- the source lifecycle status must satisfy `required_lifecycle_status`;
- the source region must match the applicable user or business context;
- the source must be accessible to the current user or system context where access controls apply.

Draft, superseded, expired, archived, or otherwise ineligible content must not be used as supporting evidence.

## 5. Retrieval and Lookup Execution

The system must execute the source according to the `usage_mode` defined in `consulting/knowledge_source_registry.csv`.

### 5.1 Deterministic sources

Sources configured for deterministic lookup must be queried through structured lookup or rule-based logic.

The LLM must not infer, reconstruct, or override deterministic business rules from narrative documents.

If a required deterministic rule cannot be found, the system must treat the result as unresolved rather than infer a likely answer.

### 5.2 RAG sources

Sources configured for RAG must be retrieved only after source eligibility has been confirmed.

Retrieval should prioritize content that:

- matches the relevant knowledge domain;
- matches the applicable region and lifecycle context;
- contains the information required to answer the user question;
- comes from the authoritative or explicitly permitted supporting source.

Only the most relevant evidence required for the answer should be passed to the LLM.

### 5.3 Strict-citation sources

For sources configured with strict citation requirements, material claims must be supported by explicit retrieved evidence from the authoritative source.

The model must not strengthen, generalize, or extrapolate beyond the wording supported by the source.

If sufficient evidence is unavailable, the system must abstain from making the claim.

## 6. Multi-Source Questions

When a query spans multiple knowledge domains, the system must:

1. identify all relevant domains;
2. resolve the authoritative source for each domain;
3. apply source eligibility checks;
4. execute each source according to its configured usage mode;
5. preserve the distinction between deterministic results and retrieved narrative evidence;
6. combine the results only after each domain has been resolved independently.

The final answer must not allow narrative evidence from one domain to override an authoritative result from another domain.

## 7. Conflict Handling

If retrieved sources conflict:

1. the primary authoritative source takes precedence over supporting sources;
2. supporting sources may add context but must not override the authoritative source;
3. deterministic structured rules take precedence over contradictory narrative statements for the same deterministic fact;
4. if authoritative sources conflict and the conflict cannot be resolved through an existing resolution rule, the system must abstain and escalate.

The LLM must not choose between unresolved authoritative conflicts based on plausibility or semantic similarity.

## 8. Abstention and Escalation

The system must abstain from providing a definitive answer when:

- no eligible authoritative source is available;
- a deterministic lookup returns no applicable rule;
- retrieved evidence is insufficient to support a material claim;
- relevant authoritative sources conflict;
- the configuration is explicitly unsupported;
- the query falls outside the documented knowledge scope;
- the applicable source requires technical, legal, or business-owner review.

Where an escalation route is defined, the system should direct the user to the appropriate function, such as Product Engineering, System Engineering, Technical Support, or Legal.

The system must not fabricate a rule, infer unsupported compatibility, or convert uncertainty into a definitive recommendation.

## 9. Output Requirements

Final answers should:

- distinguish confirmed facts from unresolved points;
- cite or identify the supporting source for material claims where required;
- preserve conditions and limitations contained in the source;
- avoid presenting assumptions as facts;
- explicitly state when evidence is insufficient;
- recommend escalation where required by this policy.

For deterministic conclusions, the answer should reflect the result returned by the structured source rather than an LLM-generated interpretation of the underlying rule.

## 10. Policy Principle

The system may use probabilistic models to understand questions, retrieve relevant evidence, and synthesize language.

It must not use probabilistic inference to replace authoritative deterministic business rules or to resolve unsupported high-risk claims.

Enterprise knowledge authority is defined by the configured business sources and governance artifacts, not by the language model.
