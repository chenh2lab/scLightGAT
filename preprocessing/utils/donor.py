from __future__ import annotations

import re

import pandas as pd


DONOR_OBS_COLUMNS = ("Sample_ID", "Donor_ID", "Group_ID")


# GSE131907 publishes Sample in the cell annotation but the biological patient
# relation in its feature-summary workbook.  Keep the reviewed mapping explicit
# so a new/unreviewed sample cannot silently be grouped by a permissive regex.
_GSE131907_PATIENT_BY_SAMPLE = {
    "LUNG_N01": "P0001",
    "LUNG_N06": "P0006",
    "LUNG_N08": "P0008",
    "LUNG_N09": "P0009",
    "LUNG_N18": "P0018",
    "LUNG_N19": "P0019",
    "LUNG_N20": "P0020",
    "LUNG_N28": "P0028",
    "LUNG_N30": "P0030",
    "LUNG_N31": "P0031",
    "LUNG_N34": "P0034",
    "LUNG_T06": "P0006",
    "LUNG_T08": "P0008",
    "LUNG_T09": "P0009",
    "LUNG_T18": "P0018",
    "LUNG_T19": "P0019",
    "LUNG_T20": "P0020",
    "LUNG_T25": "P0025",
    "LUNG_T28": "P0028",
    "LUNG_T30": "P0030",
    "LUNG_T31": "P0031",
    "LUNG_T34": "P0034",
}


# GEO GSE176078 contains one primary-tumour sample per published CID title.
_GSE176078_DONORS = {
    "CID3586",
    "CID3838",
    "CID3921",
    "CID3941",
    "CID3946",
    "CID3948",
    "CID3963",
    "CID4040",
    "CID4066",
    "CID4067",
    "CID4290A",
    "CID4398",
    "CID44041",
    "CID4461",
    "CID4463",
    "CID4465",
    "CID4471",
    "CID4495",
    "CID44971",
    "CID44991",
    "CID4513",
    "CID4515",
    "CID45171",
    "CID4523",
    "CID4530N",
    "CID4535",
}


# GSE141445's official GEO Sample_description gives this exact column suffix
# (sample number) -> patient ID relation.
_GSE141445_PATIENT_BY_SAMPLE_NUMBER = {
    "1": "patient_59",
    "2": "patient_62",
    "3": "patient_74",
    "4": "patient_72",
    "5": "patient_73",
    "6": "patient_71",
    "7": "patient_75",
    "8": "patient_77",
    "9": "patient_76",
    "10": "patient_54",
    "11": "patient_55",
    "12": "patient_56",
    "13": "patient_53",
}


def _clean_text(values: pd.Series, *, dataset: str, field: str) -> pd.Series:
    result = values.astype("object").map(
        lambda value: str(value).strip() if pd.notna(value) else pd.NA
    )
    missing = result.isna() | result.astype(str).str.lower().isin(
        {"", "<na>", "na", "n/a", "nan", "none"}
    )
    if missing.any():
        examples = values.index[missing][:5].astype(str).tolist()
        raise ValueError(
            f"{dataset}: {field} is missing for {int(missing.sum()):,} cells; "
            f"examples={examples}"
        )
    return result.astype(str)


def _source_cell_ids(obs: pd.DataFrame) -> pd.Series:
    if "source_cell_id" in obs:
        values = obs["source_cell_id"]
    else:
        values = pd.Series(obs.index.astype(str), index=obs.index)
    return values.astype(str).str.strip()


def _extract(
    values: pd.Series,
    pattern: str,
    *,
    dataset: str,
    description: str,
    group: int = 1,
) -> pd.Series:
    extracted = values.str.extract(pattern, expand=True)
    if extracted.shape[1] < group:
        raise RuntimeError(f"Invalid internal extraction group for {dataset}")
    result = extracted.iloc[:, group - 1]
    result.index = values.index
    missing = result.isna()
    if missing.any():
        examples = values.loc[missing].head(5).tolist()
        raise ValueError(
            f"{dataset}: cannot derive {description} from "
            f"{int(missing.sum()):,} source cell IDs; examples={examples}"
        )
    return result.astype(str)


def _validate_existing(
    obs: pd.DataFrame,
    column: str,
    expected: pd.Series,
    *,
    dataset: str,
) -> pd.Series:
    """Prefer an authoritative source column and verify the canonical rule."""
    if column not in obs:
        return expected.astype(str)

    observed = _clean_text(obs[column], dataset=dataset, field=column)
    mismatch = observed.ne(expected.astype(str))
    if mismatch.any():
        examples = pd.DataFrame(
            {
                "source_cell_id": _source_cell_ids(obs).loc[mismatch],
                "observed": observed.loc[mismatch],
                "expected": expected.loc[mismatch],
            }
        ).head(5)
        raise ValueError(
            f"{dataset}: source {column} conflicts with the canonical donor "
            f"mapping; examples={examples.to_dict(orient='records')}"
        )
    return observed


def _identity_for_dataset(obs: pd.DataFrame, dataset: str) -> tuple[pd.Series, pd.Series]:
    source_ids = _source_cell_ids(obs)

    if dataset == "GSE131907":
        parsed_sample = _extract(
            source_ids,
            r"_(LUNG_[NT]\d+)$",
            dataset=dataset,
            description="published lung sample",
        )
        sample = _validate_existing(
            obs,
            "Sample",
            parsed_sample,
            dataset=dataset,
        )
        donor = sample.map(_GSE131907_PATIENT_BY_SAMPLE)
        unknown = donor.isna()
        if unknown.any():
            raise ValueError(
                f"{dataset}: unreviewed Sample values: "
                f"{sorted(sample.loc[unknown].unique())}"
            )
        return sample, donor.astype(str)

    if dataset == "GSE176078":
        parsed = _extract(
            source_ids,
            r"^(CID[0-9]+[A-Z]?)_",
            dataset=dataset,
            description="published CID",
        )
        unknown = ~parsed.isin(_GSE176078_DONORS)
        if unknown.any():
            raise ValueError(
                f"{dataset}: source IDs contain CID values absent from the "
                f"official 26-sample series: {sorted(parsed.loc[unknown].unique())}"
            )
        # The source archive calls this field orig.ident.  Validate it when it
        # is present, while retaining the source-cell prefix as the patchable
        # representation shared with an already-built train.h5ad.
        if "orig.ident" in obs:
            parsed = _validate_existing(
                obs,
                "orig.ident",
                parsed,
                dataset=dataset,
            )
        return parsed, parsed.copy()

    if dataset == "GSE140228":
        if "Sample" not in obs or "Donor" not in obs:
            raise ValueError(
                f"{dataset}: official Sample and Donor columns are required; "
                "join the published Droplet/Smart-seq2 cell-info table first"
            )
        sample = _clean_text(obs["Sample"], dataset=dataset, field="Sample")
        donor = _clean_text(obs["Donor"], dataset=dataset, field="Donor")
        return sample, donor

    if dataset == "GSE132465":
        parsed = source_ids.str.extract(
            r"^((SMC\d+)-(?:T|N))_",
            expand=True,
        )
        parsed.index = source_ids.index
        if parsed.isna().any(axis=None):
            bad = source_ids.loc[parsed.iloc[:, 0].isna()].head(5).tolist()
            raise ValueError(
                f"{dataset}: cannot derive Patient/Sample from source IDs; "
                f"examples={bad}"
            )
        sample = _validate_existing(
            obs,
            "Sample",
            parsed.iloc[:, 0].astype(str),
            dataset=dataset,
        )
        donor = _validate_existing(
            obs,
            "Patient",
            parsed.iloc[:, 1].astype(str),
            dataset=dataset,
        )
        return sample, donor

    if dataset == "GSE141445":
        number = _extract(
            source_ids,
            r"-(\d+)$",
            dataset=dataset,
            description="published sample number",
        )
        donor = number.map(_GSE141445_PATIENT_BY_SAMPLE_NUMBER)
        unknown = donor.isna()
        if unknown.any():
            raise ValueError(
                f"{dataset}: unreviewed sample numbers: "
                f"{sorted(number.loc[unknown].unique())}"
            )
        if "patient" in obs:
            _validate_existing(
                obs,
                "patient",
                "prostate_" + donor.astype(str),
                dataset=dataset,
            )
        sample = "sample_" + number.astype(int).astype(str).str.zfill(2)
        return sample, donor.astype(str)

    raise ValueError(f"No canonical donor mapping is defined for {dataset}")


def validate_donor_metadata(
    obs: pd.DataFrame,
    *,
    dataset: str | None = None,
) -> None:
    missing_columns = set(DONOR_OBS_COLUMNS).difference(obs.columns)
    if missing_columns:
        raise ValueError(
            f"Missing donor metadata columns: {sorted(missing_columns)}"
        )

    for column in DONOR_OBS_COLUMNS:
        _clean_text(
            obs[column],
            dataset=dataset or "Integrated_training",
            field=column,
        )

    data_source = obs["Data_source"].astype(str)
    if dataset is not None:
        unexpected = set(data_source) - {dataset}
        if unexpected:
            raise ValueError(
                f"{dataset}: Data_source contains unexpected values "
                f"{sorted(unexpected)}"
            )

    expected_group = data_source + ":" + obs["Donor_ID"].astype(str)
    mismatch = obs["Group_ID"].astype(str).ne(expected_group)
    if mismatch.any():
        examples = pd.DataFrame(
            {
                "Data_source": data_source.loc[mismatch],
                "Donor_ID": obs.loc[mismatch, "Donor_ID"].astype(str),
                "Group_ID": obs.loc[mismatch, "Group_ID"].astype(str),
                "expected": expected_group.loc[mismatch],
            }
        ).head(5)
        raise ValueError(
            "Group_ID must equal Data_source + ':' + Donor_ID; "
            f"examples={examples.to_dict(orient='records')}"
        )

    sample_to_donor = (
        obs.assign(_sample=obs["Sample_ID"].astype(str))
        .groupby(["Data_source", "_sample"], observed=True)["Donor_ID"]
        .nunique()
    )
    conflicting = sample_to_donor[sample_to_donor.ne(1)]
    if len(conflicting):
        raise ValueError(
            "A source Sample_ID maps to multiple Donor_ID values: "
            f"{list(conflicting.index[:5])}"
        )


def attach_donor_metadata(adata, dataset: str):
    """Attach canonical sample, donor and globally namespaced group IDs.

    This is the single production resolver used by source builders, integrated
    reconstruction and the disposable migration tool.  The migration tool may
    import this module; production code must never import the migration tool.
    """
    sample, donor = _identity_for_dataset(adata.obs, dataset)
    adata.obs["Sample_ID"] = sample.to_numpy()
    adata.obs["Donor_ID"] = donor.to_numpy()
    adata.obs["Group_ID"] = (
        dataset + ":" + adata.obs["Donor_ID"].astype(str)
    )
    validate_donor_metadata(adata.obs, dataset=dataset)
    return adata
