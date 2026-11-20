import type { Detection, SelectedYoloClass } from "../types/detection";

export type DetectionPair = readonly [Detection, Detection];
export type DetectionPairDistance = (from: Detection, to: Detection) => number | null;

export const DEFAULT_CLASS_CONFIDENCE = 0.5;
export const MIN_CLASS_CONFIDENCE = 0.05;
export const MAX_CLASS_CONFIDENCE = 0.95;

export function modelClassKey(model: string, classId: number): string {
  return `${model}\u0000${classId}`;
}

export function canApplyMeasurementSettings(
  enabled: boolean,
  selectedClassCount: number,
  canEnable: boolean,
): boolean {
  return !enabled || (canEnable && selectedClassCount >= 1 && selectedClassCount <= 2);
}

export function normalizeClassConfidence(value: unknown): number {
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_CLASS_CONFIDENCE;
  return Math.min(MAX_CLASS_CONFIDENCE, Math.max(MIN_CLASS_CONFIDENCE, parsed));
}

/** 선택 클래스마다 지정된 confidence를 독립적으로 적용한다. */
export function filterDetectionsByClassConfidence(
  detections: readonly Detection[],
  selectedClasses: readonly SelectedYoloClass[],
): Detection[] {
  const confidenceByModelClass = new Map(
    selectedClasses.map((item) => [
      modelClassKey(item.model, item.id),
      normalizeClassConfidence(item.conf),
    ]),
  );
  return detections.filter((det) => {
    const threshold = confidenceByModelClass.get(modelClassKey(det.model, det.class_id));
    return threshold != null && det.conf >= threshold;
  });
}

/** 백엔드는 단일 threshold만 받으므로 선택값 중 최솟값으로 필요한 bbox를 모두 수신한다. */
export function minimumClassConfidence(selectedClasses: readonly SelectedYoloClass[]): number {
  return selectedClasses.length > 0
    ? Math.min(...selectedClasses.map((item) => normalizeClassConfidence(item.conf)))
    : DEFAULT_CLASS_CONFIDENCE;
}

/**
 * 자동 거리측정용 bbox 쌍을 만든다.
 *
 * 1개 선택 시 동종 객체끼리 각 객체의 최근접 쌍을 만들고 무순서 중복을 제거한다.
 * 2개 선택 시 첫 클래스 객체마다 두 번째 클래스의 최근접 객체를 단방향으로 잇는다.
 * class_id는 모델마다 겹칠 수 있으므로 모든 비교는 model+class_id 복합키로 수행한다.
 */
export function buildDetectionPairs(
  detections: readonly Detection[],
  selectedClasses: readonly SelectedYoloClass[],
  distanceBetween: DetectionPairDistance = (from, to) => {
    const fromX = (from.xyxy[0] + from.xyxy[2]) / 2;
    const fromY = from.xyxy[3];
    const toX = (to.xyxy[0] + to.xyxy[2]) / 2;
    const toY = to.xyxy[3];
    return Math.hypot(toX - fromX, toY - fromY);
  },
): DetectionPair[] {
  if (selectedClasses.length < 1 || selectedClasses.length > 2) return [];
  const firstKey = modelClassKey(selectedClasses[0].model, selectedClasses[0].id);
  const anchors = detections.filter(
    (det) => modelClassKey(det.model, det.class_id) === firstKey,
  );

  if (selectedClasses.length === 1) {
    const pairs: DetectionPair[] = [];
    const seen = new Set<string>();
    for (let anchorIndex = 0; anchorIndex < anchors.length; anchorIndex += 1) {
      const anchor = anchors[anchorIndex];
      let nearestIndex = -1;
      let shortest = Number.POSITIVE_INFINITY;
      for (let candidateIndex = 0; candidateIndex < anchors.length; candidateIndex += 1) {
        if (candidateIndex === anchorIndex) continue;
        const distance = distanceBetween(anchor, anchors[candidateIndex]);
        if (distance == null || !Number.isFinite(distance) || distance >= shortest) continue;
        shortest = distance;
        nearestIndex = candidateIndex;
      }
      if (nearestIndex < 0) continue;
      const pairKey = anchorIndex < nearestIndex
        ? `${anchorIndex}:${nearestIndex}`
        : `${nearestIndex}:${anchorIndex}`;
      if (seen.has(pairKey)) continue;
      seen.add(pairKey);
      pairs.push([anchor, anchors[nearestIndex]]);
    }
    return pairs;
  }

  const secondKey = modelClassKey(selectedClasses[1].model, selectedClasses[1].id);
  if (firstKey === secondKey) return [];
  const candidates = detections.filter(
    (det) => modelClassKey(det.model, det.class_id) === secondKey,
  );
  if (anchors.length === 0 || candidates.length === 0) return [];

  return anchors.flatMap((anchor): DetectionPair[] => {
    let nearest: Detection | null = null;
    let shortest = Number.POSITIVE_INFINITY;
    for (const candidate of candidates) {
      const distance = distanceBetween(anchor, candidate);
      if (distance == null || !Number.isFinite(distance) || distance >= shortest) continue;
      shortest = distance;
      nearest = candidate;
    }
    return nearest ? [[anchor, nearest]] : [];
  });
}
