import { describe, expect, it, vi } from "vitest";
import {
  canApplyMeasurementSettingsWithClasses,
  EMPTY_ENABLED_SELECTION_WARNING,
  ensureClassConfidence,
  restoreMeasurementSelection,
  shouldWarnEmptyEnabledSelection,
  swapMeasurementClassOrder,
} from "../components/MeasurementClassModal";
import {
  disableInferenceForStreamKeys,
  filterMeasurementsForActiveModel,
  makeAutoMeasurement,
  preserveWeightsResultWhileLoadingClasses,
  keepMeasurementsForActiveModel,
  readAutoMeasurementResources,
  readWeightsStatus,
  synchronizeMeasurementSelections,
} from "../pages/CamerasPage";
import type { AutoMeasurement, Detection } from "../types/detection";
import {
  buildDetectionPairs,
  canApplyMeasurementSettings,
  DEFAULT_CLASS_CONFIDENCE,
  filterDetectionsByClassConfidence,
  minimumClassConfidence,
} from "./detectionPairs";
import type { SelectedYoloClass } from "../types/detection";

function det(
  classId: number,
  name: string,
  x: number,
  conf = 0.9,
  model = "yolo26x.pt",
): Detection {
  return {
    class_id: classId,
    name,
    conf,
    xyxy: [x, 0, x + 10, 10],
    model,
  };
}

function selected(
  id: number,
  name: string,
  model: string,
  conf = 0.5,
): SelectedYoloClass {
  return { id, name, model, conf };
}

describe("buildDetectionPairs", () => {
  it("클래스 1개면 각 객체의 최근접 동종 쌍을 만들고 무순서 중복을 제거한다", () => {
    const a = det(0, "forklift", 0, 0.9, "warehouse.pt");
    const b = det(0, "forklift", 10, 0.9, "warehouse.pt");
    const c = det(0, "forklift", 100, 0.9, "warehouse.pt");

    expect(
      buildDetectionPairs(
        [a, b, c],
        [selected(0, "forklift", "warehouse.pt")],
        (from, to) => Math.abs(from.xyxy[0] - to.xyxy[0]),
      ),
    ).toEqual([[a, b], [c, b]]);
  });

  it("클래스 1개에서 유효한 world 거리가 없는 객체는 선을 만들지 않는다", () => {
    const a = det(0, "forklift", 0, 0.9, "warehouse.pt");
    const b = det(0, "forklift", 10, 0.9, "warehouse.pt");
    const c = det(0, "forklift", 30, 0.9, "warehouse.pt");

    expect(
      buildDetectionPairs(
        [a, b, c],
        [selected(0, "forklift", "warehouse.pt")],
        (from, to) => (from === b || to === b ? null : 30),
      ),
    ).toEqual([[a, c]]);
  });

  it("클래스 2개면 첫 클래스 객체마다 두 번째 클래스의 최근접 객체 1개를 잇는다", () => {
    const f1 = det(0, "forklift", 0, 0.9, "warehouse.pt");
    const f2 = det(0, "forklift", 100, 0.9, "warehouse.pt");
    const p1 = det(1, "pallet", 10, 0.9, "warehouse.pt");
    const p2 = det(1, "pallet", 80, 0.9, "warehouse.pt");

    expect(
      buildDetectionPairs(
        [f1, p1, f2, p2],
        [
          selected(0, "forklift", "warehouse.pt"),
          selected(1, "pallet", "warehouse.pt"),
        ],
        (from, to) => Math.abs(from.xyxy[0] - to.xyxy[0]),
      ),
    ).toEqual([[f1, p1], [f2, p2]]);
  });

  it("클래스 2개에서 두 번째 클래스 객체 재사용을 허용하고 역방향 쌍은 만들지 않는다", () => {
    const f1 = det(0, "forklift", 0, 0.9, "warehouse.pt");
    const f2 = det(0, "forklift", 100, 0.9, "warehouse.pt");
    const pallet = det(1, "pallet", 40, 0.9, "warehouse.pt");

    expect(
      buildDetectionPairs(
        [pallet, f2, f1],
        [
          selected(0, "forklift", "warehouse.pt"),
          selected(1, "pallet", "warehouse.pt"),
        ],
        (from, to) => Math.abs(from.xyxy[0] - to.xyxy[0]),
      ),
    ).toEqual([[f2, pallet], [f1, pallet]]);
  });

  it("선택한 활성 모델과 다른 detection은 후보에서 제외한다", () => {
    const forklift = det(0, "forklift", 0, 0.9, "warehouse.pt");
    const selectedPallet = det(1, "pallet", 40, 0.9, "warehouse.pt");
    const otherPallet = det(1, "pallet", 5, 0.9, "other.pt");

    expect(
      buildDetectionPairs(
        [forklift, otherPallet, selectedPallet],
        [
          selected(0, "forklift", "warehouse.pt"),
          selected(1, "pallet", "warehouse.pt"),
        ],
        (from, to) => Math.abs(from.xyxy[0] - to.xyxy[0]),
      ),
    ).toEqual([[forklift, selectedPallet]]);
  });

  it("선택이 0개 또는 3개면 쌍을 만들지 않는다", () => {
    const items = [det(0, "forklift", 0), det(1, "pallet", 20)];
    expect(buildDetectionPairs(items, [])).toEqual([]);
    expect(buildDetectionPairs(items, [
      selected(0, "forklift", "yolo26x.pt"),
      selected(1, "pallet", "yolo26x.pt"),
      selected(2, "box", "yolo26x.pt"),
    ])).toEqual([]);
  });
});

describe("filterDetectionsByClassConfidence", () => {
  it("선택한 클래스마다 서로 다른 confidence 임계값을 적용한다", () => {
    const forkliftLow = det(0, "forklift", 0, 0.49, "warehouse.pt");
    const forkliftAtThreshold = det(0, "forklift", 20, 0.5, "warehouse.pt");
    const palletLow = det(1, "pallet", 40, 0.79, "warehouse.pt");
    const palletHigh = det(1, "pallet", 60, 0.81, "warehouse.pt");
    const wrongModel = det(1, "pallet", 70, 0.99, "other.pt");
    const unselected = det(5, "box", 80, 0.99, "warehouse.pt");

    expect(
      filterDetectionsByClassConfidence(
        [forkliftLow, forkliftAtThreshold, palletLow, palletHigh, wrongModel, unselected],
        [
          selected(0, "forklift", "warehouse.pt", 0.5),
          selected(1, "pallet", "warehouse.pt", 0.8),
        ],
      ),
    ).toEqual([forkliftAtThreshold, palletHigh]);
  });

  it("백엔드 수집 임계값은 선택 클래스 confidence 중 최솟값을 사용한다", () => {
    expect(
      minimumClassConfidence([
        selected(0, "forklift", "warehouse.pt", 0.7),
        selected(1, "pallet", "warehouse.pt", 0.35),
      ]),
    ).toBe(0.35);
  });
});

describe("canApplyMeasurementSettings", () => {
  it("OFF는 클래스 선택 없이 적용할 수 있다", () => {
    expect(canApplyMeasurementSettings(false, 0, false)).toBe(true);
  });

  it("ON은 활성 기준점과 클래스 1개 또는 2개가 필요하고 3개째는 거부한다", () => {
    expect(canApplyMeasurementSettings(true, 0, true)).toBe(false);
    expect(canApplyMeasurementSettings(true, 1, false)).toBe(false);
    expect(canApplyMeasurementSettings(true, 1, true)).toBe(true);
    expect(canApplyMeasurementSettings(true, 2, true)).toBe(true);
    expect(canApplyMeasurementSettings(true, 3, true)).toBe(false);
  });

  it("클래스 조회가 실패해도 OFF는 적용할 수 있지만 ON은 적용할 수 없다", () => {
    expect(canApplyMeasurementSettingsWithClasses(false, 0, false, "클래스 조회 실패")).toBe(true);
    expect(canApplyMeasurementSettingsWithClasses(true, 1, true, "클래스 조회 실패")).toBe(false);
  });
});

describe("weights status compatibility", () => {
  it("이전 응답 shape가 오면 저장된 측정 선택을 지우지 않는다", () => {
    const stored: Record<string, AutoMeasurement> = {
      "ipcam-a": {
        enabled: true,
        classes: [selected(0, "forklift", "warehouse.pt", 0.75)],
        weights_uploaded_at: "2026-08-26T00:00:00Z",
      },
    };
    const oldShape = { preset_name: "yolo26x.pt", custom: null };

    expect(readWeightsStatus(oldShape)).toBeNull();
    expect(readWeightsStatus({ active_name: "best.pt" })).toBeNull();
    expect(readWeightsStatus({ active_name: "best.pt", uploaded_at: null, default_name: {} })).toBeNull();
    expect(keepMeasurementsForActiveModel(stored, oldShape)).toBe(stored);
  });

  it("유효한 응답이면 현재 활성 모델과 다른 저장 선택만 제거한다", () => {
    const stored: Record<string, AutoMeasurement> = {
      keep: {
        enabled: true,
        classes: [selected(0, "forklift", "warehouse.pt")],
        weights_uploaded_at: "2026-08-26T00:00:00Z",
      },
      remove: {
        enabled: true,
        classes: [selected(0, "person", "yolo26x.pt")],
        weights_uploaded_at: null,
      },
    };

    expect(keepMeasurementsForActiveModel(stored, {
      active_name: "warehouse.pt",
      type: "custom",
      uploaded_at: "2026-08-26T00:00:00Z",
      size_mb: 20,
      class_count: 2,
    })).toEqual({ keep: stored.keep });
  });

  it("같은 파일명이어도 uploaded_at이 다르면 저장 선택을 제거한다", () => {
    const stored: Record<string, AutoMeasurement> = {
      stale: {
        enabled: true,
        classes: [selected(7, "forklift", "best.pt")],
        weights_uploaded_at: "2026-08-26T00:00:00Z",
      },
    };

    expect(keepMeasurementsForActiveModel(stored, {
      active_name: "best.pt",
      type: "custom",
      uploaded_at: "2026-08-27T00:00:00Z",
      size_mb: 20,
      class_count: 2,
    })).toEqual({});
  });

  it("가중치 신원이 없는 구 저장값은 유효한 상태 조회 뒤 폐기한다", () => {
    const legacy: Record<string, AutoMeasurement> = {
      stale: {
        enabled: true,
        classes: [selected(0, "person", "yolo26x.pt")],
      },
    };

    expect(keepMeasurementsForActiveModel(legacy, {
      active_name: "yolo26x.pt",
      type: "preset",
      uploaded_at: null,
      size_mb: null,
      class_count: 80,
    })).toEqual({});
  });

  it("preset은 uploaded_at null까지 같을 때 저장 선택을 유지한다", () => {
    const stored: Record<string, AutoMeasurement> = {
      preset: {
        enabled: true,
        classes: [selected(0, "person", "yolo26x.pt")],
        weights_uploaded_at: null,
      },
    };

    expect(keepMeasurementsForActiveModel(stored, {
      active_name: "yolo26x.pt",
      type: "preset",
      uploaded_at: null,
      size_mb: null,
      class_count: 80,
    })).toEqual(stored);
  });

  it("새 저장값에 활성 가중치 이름과 uploaded_at 신원을 함께 기록한다", () => {
    const classes = [selected(7, "forklift", "best.pt")];

    expect(makeAutoMeasurement(true, classes, {
      active_name: "best.pt",
      type: "custom",
      uploaded_at: "2026-08-27T00:00:00Z",
      size_mb: 20,
      class_count: 2,
    })).toEqual({
      enabled: true,
      classes,
      weights_uploaded_at: "2026-08-27T00:00:00Z",
    });
  });
});

describe("class confidence reselect", () => {
  it("재선택하면 사용자가 조정한 confidence를 보존한다", () => {
    const tuned = { 4: 0.8 };

    expect(ensureClassConfidence(tuned, 4)).toBe(tuned);
    expect(ensureClassConfidence(tuned, 4)[4]).toBe(0.8);
  });

  it("처음 선택한 클래스만 기본 confidence로 초기화한다", () => {
    expect(ensureClassConfidence({}, 4)).toEqual({
      4: DEFAULT_CLASS_CONFIDENCE,
    });
  });
});

describe("empty enabled selection guidance", () => {
  it("ON인데 복원 가능한 클래스가 없으면 선택 또는 OFF 안내를 표시한다", () => {
    expect(shouldWarnEmptyEnabledSelection(true, 0)).toBe(true);
    expect(EMPTY_ENABLED_SELECTION_WARNING).toContain("OFF");
  });

  it("OFF이거나 선택된 클래스가 있으면 추가 안내가 필요 없다", () => {
    expect(shouldWarnEmptyEnabledSelection(false, 0)).toBe(false);
    expect(shouldWarnEmptyEnabledSelection(true, 1)).toBe(false);
  });
});

describe("two-class anchor order", () => {
  it("두 클래스를 맞바꾸면 기준/대상 순서가 뒤집히고 저장 순서도 그대로 유지된다", () => {
    const original = [
      selected(7, "forklift", "best.pt", 0.7),
      selected(0, "person", "best.pt", 0.4),
    ];
    const swappedIds = swapMeasurementClassOrder(original.map((item) => item.id));
    const swapped = swappedIds.map((id) => original.find((item) => item.id === id)!);

    expect(swapped.map((item) => item.name)).toEqual(["person", "forklift"]);
    expect(makeAutoMeasurement(true, swapped, {
      active_name: "best.pt",
      type: "custom",
      uploaded_at: "2026-08-27T00:00:00Z",
      size_mb: 20,
      class_count: 2,
    }).classes).toEqual(swapped);
  });

  it("클래스가 하나면 순서 교체를 하지 않는다", () => {
    const one = [7];
    expect(swapMeasurementClassOrder(one)).toBe(one);
  });
});

describe("measurement selection restore", () => {
  it("활성 모델 항목만 추린 뒤 class id 중복을 제거한다", () => {
    const wrongModelFirst = selected(0, "person", "yolo26x.pt");
    const activePerson = selected(0, "person", "best.pt", 0.7);
    const duplicateActivePerson = selected(0, "person duplicate", "best.pt", 0.2);
    const activeForklift = selected(1, "forklift", "best.pt", 0.6);

    expect(restoreMeasurementSelection(
      [wrongModelFirst, activePerson, duplicateActivePerson, activeForklift],
      "best.pt",
      [{ id: 0, name: "person" }, { id: 1, name: "forklift" }],
    )).toEqual([activePerson, activeForklift]);
  });
});

describe("measurement settings resources", () => {
  it("클래스 조회만 실패하면 가중치와 기준점 상태를 유지한 채 오류를 데이터로 돌려준다", async () => {
    const resources = await readAutoMeasurementResources(
      new Response(JSON.stringify({ enabled: true, homography: [[1, 0, 0]] }), { status: 200 }),
      new Response(JSON.stringify({
        active_name: "best.pt",
        type: "custom",
        uploaded_at: "2026-08-27T00:00:00Z",
        size_mb: 20,
        class_count: 2,
      }), { status: 200 }),
      new Response("unavailable", { status: 503 }),
    );

    expect(resources.canEnable).toBe(true);
    expect(resources.weights.active_name).toBe("best.pt");
    expect(resources.classes).toEqual([]);
    expect(resources.classesError).toContain("클래스");
  });
});

describe("dropped measurement synchronization", () => {
  it("가중치 신원이 달라 폐기한 카메라 키를 함께 반환한다", () => {
    const current: Record<string, AutoMeasurement> = {
      keep: {
        enabled: true,
        classes: [selected(0, "person", "best.pt")],
        weights_uploaded_at: "new",
      },
      drop: {
        enabled: true,
        classes: [selected(1, "forklift", "best.pt")],
        weights_uploaded_at: "old",
      },
    };

    expect(filterMeasurementsForActiveModel(current, {
      active_name: "best.pt",
      type: "custom",
      uploaded_at: "new",
      size_mb: 20,
      class_count: 2,
    })).toEqual({ kept: { keep: current.keep }, droppedStreamKeys: ["drop"] });
  });

  it("폐기한 각 카메라의 백엔드 추론을 OFF로 동기화한다", async () => {
    const request = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }));

    await expect(disableInferenceForStreamKeys(["ipcam-a", "ipcam-b"], request)).resolves.toEqual([]);
    expect(request).toHaveBeenCalledTimes(2);
    expect(request).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining("/api/ipcams/ipcam-a/inference"),
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: false }),
      },
    );
  });

  it("전체 선택 초기화도 기존 모든 카메라를 백엔드 OFF로 동기화한다", async () => {
    const current: Record<string, AutoMeasurement> = {
      "ipcam-a": {
        enabled: true,
        classes: [selected(0, "person", "best.pt")],
        weights_uploaded_at: "old",
      },
      "ipcam-b": {
        enabled: true,
        classes: [selected(1, "forklift", "best.pt")],
        weights_uploaded_at: "old",
      },
    };
    const request = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }));

    await expect(synchronizeMeasurementSelections(current, {}, request)).resolves.toEqual({
      droppedStreamKeys: ["ipcam-a", "ipcam-b"],
      failedStreamKeys: [],
    });
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("이미 삭제된 카메라의 404는 OFF 완료로 취급한다", async () => {
    const request = vi.fn().mockResolvedValue(new Response("not found", { status: 404 }));

    await expect(disableInferenceForStreamKeys(["deleted-cam"], request)).resolves.toEqual([]);
  });
});

describe("weights refresh error priority", () => {
  it("클래스 재조회 실패는 가중치 변경 결과를 예외로 덮어쓰지 않는다", async () => {
    await expect(preserveWeightsResultWhileLoadingClasses(
      async () => { throw new Error("클래스 조회 실패"); },
    )).resolves.toEqual([]);
  });
});
