export interface Detection {
  class_id: number;
  name: string;
  conf: number;
  xyxy: [number, number, number, number];
  model: string;
}

export interface YoloClass {
  id: number;
  name: string;
}

export interface SelectedYoloClass extends YoloClass {
  model: string;
  conf: number;
}

export interface AutoMeasurement {
  enabled: boolean;
  classes: SelectedYoloClass[];
  weights_uploaded_at?: string | null;
}

export interface WeightsStatus {
  active_name: string;
  type: "preset" | "custom";
  uploaded_at: string | null;
  size_mb: number | null;
  class_count: number;
  refresh_error?: string;
}
