/** Real wide-schema tool for the Pi regression in #641. */
import { appendFileSync } from "node:fs";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.registerTool(defineTool({
    name: "todo",
    label: "Todo",
    description: "Update the status of an existing task.",
    parameters: Type.Object({
      action: Type.String({ enum: ["update"] }),
      subject: Type.Optional(Type.String()),
      description: Type.Optional(Type.String()),
      activeForm: Type.Optional(Type.String()),
      status: Type.Optional(Type.String({ enum: ["pending", "in_progress", "completed"] })),
      blockedBy: Type.Optional(Type.Array(Type.Integer())),
      addBlockedBy: Type.Optional(Type.Array(Type.Integer())),
      removeBlockedBy: Type.Optional(Type.Array(Type.Integer())),
      owner: Type.Optional(Type.String()),
      metadata: Type.Optional(Type.Object({})),
      id: Type.Optional(Type.Integer()),
      includeDeleted: Type.Optional(Type.Boolean()),
    }),
    async execute(_id, params) {
      appendFileSync("todo-calls.jsonl", JSON.stringify(params) + "\n");
      const unexpected = Object.keys(params).some(name =>
        !["action", "id", "status"].includes(name));
      if (params.action !== "update" || !Number.isInteger(params.id) || !params.status ||
          unexpected) {
        throw new Error("Update requires action, id and status, with no other changes.");
      }
      return {
        content: [{ type: "text", text: `Task ${params.id}: ${params.status}` }],
        details: params,
      };
    },
  }));
}
