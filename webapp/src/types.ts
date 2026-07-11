export interface BlockData {
  index: number;
  kind: string;
  mixer_class: string;
  mlp_class: string | null;
  params: number;
  detail: Record<string, unknown>;
}

export interface ExtraData {
  name: string;
  kind: string;
  class: string;
  params: number;
  detail: Record<string, unknown>;
}

export interface ModelData {
  id: string;
  name: string;
  description: string;
  config: Record<string, unknown>;
  total_params: number;
  blocks: BlockData[];
  extras: ExtraData[];
}

export interface ModelsFile {
  models: ModelData[];
}
