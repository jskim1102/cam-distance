import { renderToStaticMarkup } from "react-dom/server";
import { expect, it, vi } from "vitest";
import MeasurementClassModal from "./MeasurementClassModal";

it.each([
  ["yolo26n.pt", "기본(yolo26n.pt)으로 되돌리기"],
  [undefined, "기본으로 되돌리기"],
])("기본 모델 이름 %s에 맞는 복원 버튼을 표시한다", (name, label) => {
  const html = renderToStaticMarkup(
    <MeasurementClassModal
      open cameraName="카메라" classes={[]} initialEnabled={false} initialSelection={[]}
      canEnable saving={false} weightsBusy={false} weightsError="" weightsNotice=""
      classesError="" selectionResetToken={0}
      weights={{
        active_name: "custom.pt", default_name: name, type: "custom",
        uploaded_at: "2026-11-20", size_mb: 1, class_count: 1,
      }}
      onClose={vi.fn()} onConfirm={vi.fn()} onUploadWeights={vi.fn()} onResetWeights={vi.fn()}
    />,
  );
  expect(html).toContain(label);
});
