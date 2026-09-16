# Data provenance and reuse

## File

`Database_86_with_DOI.xlsx` is the frozen dataset used for the final manuscript analysis.

- Complete records used by the model: **86**
- Kinetic targets: **Km, kcat, Vmax**
- Reconstructed publication/source groups: **23**
- Stable grouping field used by the analysis: `publication_id`
- Primary worksheet: `Matched_86`

The DOI/publication identifiers were reconstructed to enable publication-aware validation without changing the 86-record analytical cohort.

## Provenance

The records were curated from **NanozymeDB** and the scientific publications referenced by those records. NanozymeDB is described by Sharma et al., *Journal of Nanotechnology Research* (2023), DOI: `10.26502/jnr.2688-85210039`.

The NanozymeDB article describes the resource as an open-source database and is published under **CC BY 4.0**. Individual kinetic measurements, bibliographic metadata, and source-publication content may also be subject to the terms and rights of their original sources.

## License note

The root MIT License applies to the software in this repository. **This curated data file is not relicensed under MIT.** No ownership is claimed over the underlying source measurements.

Users should:

1. cite the associated manuscript/repository;
2. cite NanozymeDB and the relevant source publications when reusing individual records;
3. verify any source-specific reuse conditions for their intended application.

## Why publication IDs matter

Multiple records can originate from the same paper and therefore share experimental context such as synthesis procedures, assay conditions, substrates, and laboratory-specific practices. Random row-wise cross-validation can place related records in both training and validation partitions. The final analysis therefore also uses `GroupKFold` so that all records from a publication group are held out together.
