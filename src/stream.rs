use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::message::ChatMessage;
use crate::response::FinishReason;
use crate::usage::Usage;

/// Incremental stream event. Provider identity is not exposed here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamChunk {
    ContentDelta {
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: Option<String>,
    },
    Usage(Usage),
    Finished {
        reason: FinishReason,
        message: Option<ChatMessage>,
    },
}

/// Provider-neutral stream of [`StreamChunk`] values.
pub struct ChatStream {
    inner: Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>,
    /// Wall time from stream construction until the consumer drops or finishes.
    started: std::time::Instant,
}

impl ChatStream {
    pub fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<StreamChunk>> + Send + 'static,
    {
        Self {
            inner: Box::pin(stream),
            started: std::time::Instant::now(),
        }
    }

    pub fn empty() -> Self {
        Self::from_stream(futures::stream::empty())
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

impl Stream for ChatStream {
    type Item = Result<StreamChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl std::fmt::Debug for ChatStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatStream")
            .field("elapsed", &self.elapsed())
            .finish_non_exhaustive()
    }
}
