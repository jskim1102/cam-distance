import { useEffect, useMemo, useState } from "react";
import type { SelectedYoloClass, WeightsStatus, YoloClass } from "../types/detection";
import {
  canApplyMeasurementSettings,
  DEFAULT_CLASS_CONFIDENCE,
  MAX_CLASS_CONFIDENCE,
  MIN_CLASS_CONFIDENCE,
  normalizeClassConfidence,
} from "../utils/detectionPairs";
import Modal from "./Modal";

interface Props {
  open: boolean;
  cameraName: string;
  classes: YoloClass[];
  initialEnabled: boolean;
  initialSelection: SelectedYoloClass[];
  canEnable: boolean;
  saving: boolean;
  weights: WeightsStatus | null;
  weightsBusy: boolean;
  weightsError: string;
  weightsNotice: string;
  classesError: string;
  selectionResetToken: number;
  onClose: () => void;
  onConfirm: (enabled: boolean, classes: SelectedYoloClass[]) => void;
  onUploadWeights: (file: File) => void;
  onResetWeights: () => void;
}

export function ensureClassConfidence(
  current: Record<number, number>,
  classId: number,
): Record<number, number> {
  if (Object.prototype.hasOwnProperty.call(current, classId)) return current;
  return { ...current, [classId]: DEFAULT_CLASS_CONFIDENCE };
}

export const EMPTY_ENABLED_SELECTION_WARNING =
  "선택된 클래스가 없습니다. 클래스를 선택하거나 자동 거리측정을 OFF로 변경하세요.";

export function shouldWarnEmptyEnabledSelection(
  enabled: boolean,
  selectedCount: number,
): boolean {
  return enabled && selectedCount === 0;
}

export function restoreMeasurementSelection(
  initialSelection: SelectedYoloClass[],
  activeName: string | null,
  classes: YoloClass[],
): SelectedYoloClass[] {
  if (activeName == null) return [];
  const activeSelection = initialSelection.filter((item) => item.model === activeName);
  return activeSelection.filter((item, index, all) => (
    classes.some((candidate) => candidate.id === item.id)
    && all.findIndex((candidate) => candidate.id === item.id) === index
  )).slice(0, 2);
}

export function swapMeasurementClassOrder(selectedIds: number[]): number[] {
  return selectedIds.length === 2 ? [selectedIds[1], selectedIds[0]] : selectedIds;
}

export function canApplyMeasurementSettingsWithClasses(
  enabled: boolean,
  selectedCount: number,
  canEnable: boolean,
  classesError: string,
): boolean {
  return canApplyMeasurementSettings(enabled, selectedCount, canEnable)
    && (!enabled || classesError === "");
}

export default function MeasurementClassModal({
  open,
  cameraName,
  classes,
  initialEnabled,
  initialSelection,
  canEnable,
  saving,
  weights,
  weightsBusy,
  weightsError,
  weightsNotice,
  classesError,
  selectionResetToken,
  onClose,
  onConfirm,
  onUploadWeights,
  onResetWeights,
}: Props) {
  const [query, setQuery] = useState("");
  const [enabled, setEnabled] = useState(false);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [confidenceById, setConfidenceById] = useState<Record<number, number>>({});
  const [weightFile, setWeightFile] = useState<File | null>(null);

  const activeName = weights?.active_name ?? null;
  const customWeightsActive = weights?.type === "custom";

  useEffect(() => {
    if (!open) return;
    const restored = restoreMeasurementSelection(initialSelection, activeName, classes);
    setQuery("");
    setEnabled(initialEnabled);
    setSelectedIds(restored.map((item) => item.id));
    setConfidenceById(Object.fromEntries(
      restored.map((item) => [item.id, normalizeClassConfidence(item.conf)]),
    ));
    setWeightFile(null);
  }, [
    activeName,
    classes,
    initialEnabled,
    initialSelection,
    open,
    selectionResetToken,
  ]);

  const locked = saving || weightsBusy;
  const classSelectionLocked = locked || classesError !== "";
  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return needle ? classes.filter((item) => item.name.toLowerCase().includes(needle)) : classes;
  }, [classes, query]);
  const selectedClasses: SelectedYoloClass[] = activeName == null
    ? []
    : selectedIds.flatMap((id) => {
        const selected = classes.find((item) => item.id === id);
        return selected
          ? [{
              ...selected,
              model: activeName,
              conf: normalizeClassConfidence(confidenceById[id]),
            }]
          : [];
      });
  const canApply = canApplyMeasurementSettingsWithClasses(
    enabled,
    selectedClasses.length,
    canEnable,
    classesError,
  );

  const confirm = () => {
    if (canApply) onConfirm(enabled, selectedClasses);
  };

  return (
    <Modal
      open={open}
      onClose={locked ? () => {} : onClose}
      title={`${cameraName} · 자동 측정 설정`}
      maxWidth={620}
    >
      <section className="measure-weights" aria-labelledby="measure-weights-title">
        <div className="measure-weights-head">
          <div>
            <strong id="measure-weights-title">추론 가중치</strong>
            <p>활성 가중치 하나를 모든 카메라에 적용합니다.</p>
          </div>
          <span className={`measure-weight-badge${customWeightsActive ? " custom" : ""}`}>
            {weights?.type === "custom" ? "CUSTOM" : "PRESET"}
          </span>
        </div>
        <div className="measure-weight-stack">
          <div className="measure-weight-current">
            <span>ACTIVE</span>
            <strong>{weights?.active_name ?? "불러오는 중…"}</strong>
            <small>
              클래스 {weights?.class_count ?? "—"}개
              {weights?.size_mb != null ? ` · ${weights.size_mb.toFixed(2)} MB` : ""}
            </small>
          </div>
        </div>
        <div className="measure-weight-upload">
          <input
            key={selectionResetToken}
            type="file"
            accept=".pt"
            disabled={locked}
            aria-label="YOLO .pt 가중치 파일"
            onChange={(event) => setWeightFile(event.target.files?.[0] ?? null)}
          />
          <button
            type="button"
            className="btn sm"
            disabled={locked || weightFile == null}
            onClick={() => weightFile && onUploadWeights(weightFile)}
          >
            {weightsBusy ? "업로드 중…" : "업로드"}
          </button>
        </div>
        {customWeightsActive && (
          <button
            type="button"
            className="btn sm measure-weight-reset"
            disabled={locked}
            onClick={onResetWeights}
          >
            기본{weights.default_name ? `(${weights.default_name})` : ""}으로 되돌리기
          </button>
        )}
        {weightsBusy && (
          <p className="measure-weight-progress" role="status">
            가중치를 검증하고 적용하는 중입니다…
          </p>
        )}
        {weightsError && <p className="measure-weight-error" role="alert">{weightsError}</p>}
        {weightsNotice && <p className="measure-weight-notice" role="status">{weightsNotice}</p>}
      </section>

      <div className="measure-state-control">
        <div>
          <strong>자동 거리측정</strong>
          <span className={`measure-state-label ${enabled ? "on" : "off"}`}>
            {enabled ? "ON" : "OFF"}
          </span>
        </div>
        <div className="measure-state-buttons" role="group" aria-label="자동 거리측정 상태">
          <button
            type="button"
            className={`btn sm${!enabled ? " selected" : ""}`}
            disabled={locked}
            aria-pressed={!enabled}
            onClick={() => setEnabled(false)}
          >
            OFF
          </button>
          <button
            type="button"
            className={`btn sm${enabled ? " selected on" : ""}`}
            disabled={classSelectionLocked || !canEnable}
            aria-pressed={enabled}
            onClick={() => setEnabled(true)}
          >
            ON
          </button>
        </div>
      </div>
      {!canEnable && (
        <p className="measure-state-warning">ON으로 설정하려면 기준점을 저장하고 측정을 활성화하세요.</p>
      )}
      {shouldWarnEmptyEnabledSelection(enabled, selectedClasses.length) && (
        <p className="measure-state-warning">{EMPTY_ENABLED_SELECTION_WARNING}</p>
      )}
      <p className="measure-class-help">
        1개 또는 2개를 선택하세요. 2개면 먼저 고른 기준 클래스의 각 객체에서 가장 가까운
        두 번째 클래스까지 선을 1개 그립니다.
      </p>
      <div className="field">
        <label htmlFor="measure-class-search">클래스 검색</label>
        <input
          id="measure-class-search"
          value={query}
          disabled={classSelectionLocked}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="person, forklift, pallet ..."
          autoFocus
        />
      </div>
      <div className="measure-class-summary" role="status">
        <span>선택 {selectedClasses.length}/2</span>
        {selectedClasses.length === 2 ? (
          <>
            <span className="measure-class-chip anchor">
              <small>기준</small>{selectedClasses[0].name}
            </span>
            <span className="measure-class-direction" aria-hidden="true">→</span>
            <span className="measure-class-chip">{selectedClasses[1].name}</span>
            <button
              type="button"
              className="btn sm measure-class-swap"
              disabled={classSelectionLocked}
              aria-label="기준 클래스와 대상 클래스 순서 바꾸기"
              onClick={() => setSelectedIds((current) => swapMeasurementClassOrder(current))}
            >
              순서 교체
            </button>
          </>
        ) : selectedClasses.map((item) => (
          <span className="measure-class-chip" key={item.id}>{item.name}</span>
        ))}
      </div>

      <div className="measure-confidence-list" aria-label="클래스별 confidence 설정">
        {selectedClasses.map((item) => (
          <label className="measure-confidence-row" key={item.id}>
            <span className="measure-confidence-name">{item.name}</span>
            <input
              type="range"
              min={MIN_CLASS_CONFIDENCE}
              max={MAX_CLASS_CONFIDENCE}
              step="0.05"
              value={item.conf}
              disabled={classSelectionLocked}
              onChange={(event) => setConfidenceById((current) => ({
                ...current,
                [item.id]: normalizeClassConfidence(Number(event.target.value)),
              }))}
              aria-label={`${item.name} confidence`}
            />
            <output>{item.conf.toFixed(2)}</output>
          </label>
        ))}
        {selectedClasses.length === 0 && (
          <p className="measure-class-empty">클래스를 선택하면 confidence를 조정할 수 있습니다.</p>
        )}
      </div>

      <div className="measure-class-grid" aria-label="활성 YOLO 클래스 목록">
        {classesError !== "" && (
          <p className="measure-class-fetch-error" role="alert">{classesError}</p>
        )}
        {classesError === "" && visible.map((item) => {
          const checked = selectedIds.includes(item.id);
          return (
            <label className={`measure-class-option${checked ? " selected" : ""}`} key={item.id}>
              <input
                type="checkbox"
                checked={checked}
                disabled={classSelectionLocked || (!checked && selectedIds.length >= 2)}
                onChange={() => {
                  if (checked) {
                    setSelectedIds((current) => current.filter((id) => id !== item.id));
                    return;
                  }
                  setSelectedIds((current) => (
                    current.length >= 2 ? current : [...current, item.id]
                  ));
                  setConfidenceById((current) => ensureClassConfidence(current, item.id));
                }}
              />
              <span>{item.name}</span>
            </label>
          );
        })}
        {classesError === "" && visible.length === 0 && (
          <p className="measure-class-empty">일치하는 클래스가 없습니다.</p>
        )}
      </div>

      <div className="modal-actions">
        <button type="button" className="btn" disabled={locked} onClick={onClose}>취소</button>
        <button
          type="button"
          className="btn primary"
          disabled={locked || !canApply}
          onClick={confirm}
        >
          {saving ? "저장 중…" : "적용"}
        </button>
      </div>
    </Modal>
  );
}
