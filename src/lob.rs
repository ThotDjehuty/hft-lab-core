//! Limit Order Book (LOB) implementation in Rust
//! 
//! Features:
//! - Multi-level orderbook capture and storage
//! - Orderbook snapshot and diff reconstruction
//! - Real-time LOB analytics (spread, depth, imbalance)
//! - High-performance orderbook processing
//! 
//! Inspired by: https://github.com/pfei-sa/binance-LOB

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[cfg(feature = "python")]
use chrono::Utc;

/// Single level in the order book
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

impl OrderBookLevel {
    pub fn new(price: f64, quantity: f64) -> Self {
        OrderBookLevel { price, quantity }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl OrderBookLevel {
    #[new]
    fn py_new(price: f64, quantity: f64) -> Self {
        Self::new(price, quantity)
    }
    
    fn __repr__(&self) -> String {
        format!("OrderBookLevel(price={:.4}, quantity={:.4})", self.price, self.quantity)
    }
}

/// Full orderbook snapshot
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub timestamp: String,
    pub symbol: String,
    pub last_update_id: u64,
    pub exchange: String,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

impl OrderBookSnapshot {
    pub fn new(
        timestamp: String,
        symbol: String,
        last_update_id: u64,
        exchange: String,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    ) -> Self {
        OrderBookSnapshot {
            timestamp,
            symbol,
            last_update_id,
            exchange,
            bids,
            asks,
        }
    }
    
    /// Get best bid and ask
    pub fn top_of_book(&self) -> (f64, f64) {
        let bid = self.bids.first().map(|(p, _)| *p).unwrap_or(0.0);
        let ask = self.asks.first().map(|(p, _)| *p).unwrap_or(0.0);
        (bid, ask)
    }
    
    /// Get mid price
    pub fn mid_price(&self) -> f64 {
        let (bid, ask) = self.top_of_book();
        if bid > 0.0 && ask > 0.0 {
            (bid + ask) / 2.0
        } else {
            0.0
        }
    }
    
    /// Get spread
    pub fn spread(&self) -> f64 {
        let (bid, ask) = self.top_of_book();
        if bid > 0.0 && ask > 0.0 {
            ask - bid
        } else {
            0.0
        }
    }
    
    /// Get spread in basis points
    pub fn spread_bps(&self) -> f64 {
        let spread = self.spread();
        let mid = self.mid_price();
        if mid > 0.0 {
            (spread / mid) * 10000.0
        } else {
            0.0
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl OrderBookSnapshot {
    #[new]
    fn py_new(
        timestamp: String,
        symbol: String,
        last_update_id: u64,
        exchange: String,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    ) -> Self {
        Self::new(timestamp, symbol, last_update_id, exchange, bids, asks)
    }
    
    #[pyo3(name = "top_of_book")]
    fn py_top_of_book(&self) -> (f64, f64) { self.top_of_book() }
    
    #[pyo3(name = "mid_price")]
    fn py_mid_price(&self) -> f64 { self.mid_price() }
    
    #[pyo3(name = "spread")]
    fn py_spread(&self) -> f64 { self.spread() }
    
    #[pyo3(name = "spread_bps")]
    fn py_spread_bps(&self) -> f64 { self.spread_bps() }
    
    fn __repr__(&self) -> String {
        format!(
            "OrderBookSnapshot(symbol={}, timestamp={}, bids={}, asks={}, spread_bps={:.2})",
            self.symbol,
            self.timestamp,
            self.bids.len(),
            self.asks.len(),
            self.spread_bps()
        )
    }
}

/// Differential orderbook update
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderBookUpdate {
    pub timestamp: String,
    pub symbol: String,
    pub first_update_id: u64,
    pub final_update_id: u64,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

impl OrderBookUpdate {
    pub fn new(
        timestamp: String,
        symbol: String,
        first_update_id: u64,
        final_update_id: u64,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    ) -> Self {
        OrderBookUpdate {
            timestamp,
            symbol,
            first_update_id,
            final_update_id,
            bids,
            asks,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl OrderBookUpdate {
    #[new]
    fn py_new(
        timestamp: String,
        symbol: String,
        first_update_id: u64,
        final_update_id: u64,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    ) -> Self {
        Self::new(timestamp, symbol, first_update_id, final_update_id, bids, asks)
    }
}

/// LOB Analytics computed from orderbook
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LOBAnalytics {
    pub timestamp: String,
    pub symbol: String,
    
    // Spread metrics
    pub best_bid: f64,
    pub best_ask: f64,
    pub spread_abs: f64,
    pub spread_bps: f64,
    pub mid_price: f64,
    
    // Depth metrics
    pub bid_depth_1: f64,
    pub ask_depth_1: f64,
    pub bid_depth_5: f64,
    pub ask_depth_5: f64,
    pub bid_depth_10: f64,
    pub ask_depth_10: f64,
    
    // Imbalance metrics
    pub volume_imbalance: f64,
    pub price_imbalance: f64,
    pub depth_imbalance_1: f64,
    pub depth_imbalance_5: f64,
    
    // Book shape
    pub bid_levels: usize,
    pub ask_levels: usize,
    pub total_bid_volume: f64,
    pub total_ask_volume: f64,
    
    // Liquidity metrics
    pub effective_spread_bps: f64,
    pub market_impact_10k: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl LOBAnalytics {
    fn __repr__(&self) -> String {
        format!(
            "LOBAnalytics(symbol={}, spread_bps={:.2}, imbalance={:.2}%, impact={:.2}bps)",
            self.symbol,
            self.spread_bps,
            self.volume_imbalance * 100.0,
            self.market_impact_10k
        )
    }
}

/// Calculate depth within percentage of best price
fn calc_depth(levels: &[(f64, f64)], reference_price: f64, pct: f64, is_bid: bool) -> f64 {
    let threshold = if is_bid {
        reference_price * (1.0 - pct / 100.0)
    } else {
        reference_price * (1.0 + pct / 100.0)
    };
    
    let mut depth = 0.0;
    for (price, qty) in levels {
        if is_bid {
            if *price >= threshold {
                depth += price * qty; // Dollar volume
            } else {
                break;
            }
        } else {
            if *price <= threshold {
                depth += price * qty;
            } else {
                break;
            }
        }
    }
    depth
}

/// Calculate market impact for a given dollar amount
fn calc_market_impact(levels: &[(f64, f64)], target_dollars: f64) -> f64 {
    let mut remaining = target_dollars;
    let mut total_cost = 0.0;
    let mut total_qty = 0.0;
    
    for (price, qty) in levels {
        let level_dollars = price * qty;
        if remaining >= level_dollars {
            total_cost += level_dollars;
            total_qty += qty;
            remaining -= level_dollars;
        } else {
            let qty_needed = remaining / price;
            total_cost += remaining;
            total_qty += qty_needed;
            break;
        }
        
        if remaining <= 0.0 {
            break;
        }
    }
    
    if total_qty > 0.0 {
        total_cost / total_qty
    } else {
        0.0
    }
}

/// Calculate comprehensive LOB analytics
#[cfg_attr(feature = "python", pyfunction)]
pub fn calculate_lob_analytics(snapshot: &OrderBookSnapshot) -> LOBAnalytics {
    if snapshot.bids.is_empty() || snapshot.asks.is_empty() {
        return LOBAnalytics {
            timestamp: snapshot.timestamp.clone(),
            symbol: snapshot.symbol.clone(),
            best_bid: 0.0,
            best_ask: 0.0,
            spread_abs: 0.0,
            spread_bps: 0.0,
            mid_price: 0.0,
            bid_depth_1: 0.0,
            ask_depth_1: 0.0,
            bid_depth_5: 0.0,
            ask_depth_5: 0.0,
            bid_depth_10: 0.0,
            ask_depth_10: 0.0,
            volume_imbalance: 0.0,
            price_imbalance: 0.0,
            depth_imbalance_1: 0.0,
            depth_imbalance_5: 0.0,
            bid_levels: 0,
            ask_levels: 0,
            total_bid_volume: 0.0,
            total_ask_volume: 0.0,
            effective_spread_bps: 0.0,
            market_impact_10k: 0.0,
        };
    }
    
    // Basic prices
    let best_bid = snapshot.bids[0].0;
    let best_ask = snapshot.asks[0].0;
    let mid_price = (best_bid + best_ask) / 2.0;
    let spread_abs = best_ask - best_bid;
    let spread_bps = (spread_abs / mid_price) * 10000.0;
    
    // Calculate depth at different levels
    let bid_depth_1 = calc_depth(&snapshot.bids, best_bid, 0.1, true);
    let ask_depth_1 = calc_depth(&snapshot.asks, best_ask, 0.1, false);
    let bid_depth_5 = calc_depth(&snapshot.bids, best_bid, 0.5, true);
    let ask_depth_5 = calc_depth(&snapshot.asks, best_ask, 0.5, false);
    let bid_depth_10 = calc_depth(&snapshot.bids, best_bid, 1.0, true);
    let ask_depth_10 = calc_depth(&snapshot.asks, best_ask, 1.0, false);
    
    // Total volumes
    let total_bid_volume: f64 = snapshot.bids.iter().map(|(p, q)| p * q).sum();
    let total_ask_volume: f64 = snapshot.asks.iter().map(|(p, q)| p * q).sum();
    
    // Imbalance metrics
    let volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume + 1e-10);
    let depth_imbalance_1 = (bid_depth_1 - ask_depth_1) / (bid_depth_1 + ask_depth_1 + 1e-10);
    let depth_imbalance_5 = (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5 + 1e-10);
    
    // Price-weighted imbalance
    let bid_sum: f64 = snapshot.bids.iter().map(|(_, q)| q).sum();
    let ask_sum: f64 = snapshot.asks.iter().map(|(_, q)| q).sum();
    let bid_wavg = if bid_sum > 0.0 {
        snapshot.bids.iter().map(|(p, q)| p * q).sum::<f64>() / bid_sum
    } else {
        0.0
    };
    let ask_wavg = if ask_sum > 0.0 {
        snapshot.asks.iter().map(|(p, q)| p * q).sum::<f64>() / ask_sum
    } else {
        0.0
    };
    let price_imbalance = (bid_wavg - ask_wavg) / mid_price;
    
    // Market impact for $10k order
    let avg_buy_price = calc_market_impact(&snapshot.asks, 10000.0);
    let avg_sell_price = calc_market_impact(&snapshot.bids, 10000.0);
    
    let (effective_spread_bps, market_impact_10k) = if avg_buy_price > 0.0 && avg_sell_price > 0.0 {
        let effective_spread = avg_buy_price - avg_sell_price;
        let eff_spread_bps = (effective_spread / mid_price) * 10000.0;
        let impact = ((avg_buy_price - best_ask) / best_ask + (best_bid - avg_sell_price) / best_bid) / 2.0 * 10000.0;
        (eff_spread_bps, impact)
    } else {
        (spread_bps, 0.0)
    };
    
    LOBAnalytics {
        timestamp: snapshot.timestamp.clone(),
        symbol: snapshot.symbol.clone(),
        best_bid,
        best_ask,
        spread_abs,
        spread_bps,
        mid_price,
        bid_depth_1,
        ask_depth_1,
        bid_depth_5,
        ask_depth_5,
        bid_depth_10,
        ask_depth_10,
        volume_imbalance,
        price_imbalance,
        depth_imbalance_1,
        depth_imbalance_5,
        bid_levels: snapshot.bids.len(),
        ask_levels: snapshot.asks.len(),
        total_bid_volume,
        total_ask_volume,
        effective_spread_bps,
        market_impact_10k,
    }
}

/// Apply differential update to orderbook snapshot
#[cfg_attr(feature = "python", pyfunction)]
pub fn apply_orderbook_update(
    snapshot: &OrderBookSnapshot,
    update: &OrderBookUpdate,
    max_levels: usize,
) -> OrderBookSnapshot {
    // Convert to BTreeMap for efficient updates
    let mut bid_map: BTreeMap<ordered_float::OrderedFloat<f64>, f64> = snapshot
        .bids
        .iter()
        .map(|(p, q)| (ordered_float::OrderedFloat(*p), *q))
        .collect();
    
    let mut ask_map: BTreeMap<ordered_float::OrderedFloat<f64>, f64> = snapshot
        .asks
        .iter()
        .map(|(p, q)| (ordered_float::OrderedFloat(*p), *q))
        .collect();
    
    // Apply bid updates
    for (price, qty) in &update.bids {
        let key = ordered_float::OrderedFloat(*price);
        if *qty == 0.0 {
            bid_map.remove(&key);
        } else {
            bid_map.insert(key, *qty);
        }
    }
    
    // Apply ask updates
    for (price, qty) in &update.asks {
        let key = ordered_float::OrderedFloat(*price);
        if *qty == 0.0 {
            ask_map.remove(&key);
        } else {
            ask_map.insert(key, *qty);
        }
    }
    
    // Convert back to sorted vectors
    let bids: Vec<(f64, f64)> = bid_map
        .iter()
        .rev() // Bids: highest first
        .take(max_levels)
        .map(|(p, q)| (p.into_inner(), *q))
        .collect();
    
    let asks: Vec<(f64, f64)> = ask_map
        .iter()
        .take(max_levels)
        .map(|(p, q)| (p.into_inner(), *q))
        .collect();
    
    OrderBookSnapshot {
        timestamp: update.timestamp.clone(),
        symbol: update.symbol.clone(),
        last_update_id: update.final_update_id,
        exchange: snapshot.exchange.clone(),
        bids,
        asks,
    }
}

/// Parse Binance orderbook response to OrderBookSnapshot
#[cfg(feature = "python")]
#[pyfunction]
pub fn parse_binance_orderbook(
    data: &Bound<'_, PyDict>,
    symbol: String,
    exchange: String,
) -> PyResult<OrderBookSnapshot> {
    let timestamp = Utc::now().to_rfc3339();
    
    let last_update_id = data
        .get_item("lastUpdateId")?
        .and_then(|v| v.extract::<u64>().ok())
        .unwrap_or(0);
    
    let bids = if let Some(bids_list) = data.get_item("bids")? {
        parse_price_levels(&bids_list)?
    } else {
        Vec::new()
    };
    
    let asks = if let Some(asks_list) = data.get_item("asks")? {
        parse_price_levels(&asks_list)?
    } else {
        Vec::new()
    };
    
    Ok(OrderBookSnapshot::new(
        timestamp,
        symbol,
        last_update_id,
        exchange,
        bids,
        asks,
    ))
}

/// Helper to parse price levels from Python list
#[cfg(feature = "python")]
fn parse_price_levels(list: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    let py_list = list.downcast::<PyList>()?;
    let mut levels = Vec::new();
    
    for item in py_list.iter() {
        if let Ok(level_list) = item.downcast::<PyList>() {
            if level_list.len() >= 2 {
                let price = level_list.get_item(0)?.extract::<String>()?.parse::<f64>().unwrap_or(0.0);
                let qty = level_list.get_item(1)?.extract::<String>()?.parse::<f64>().unwrap_or(0.0);
                if price > 0.0 && qty > 0.0 {
                    levels.push((price, qty));
                }
            }
        }
    }
    
    Ok(levels)
}

/// Register LOB functions with Python module
#[cfg(feature = "python")]
pub fn register_lob_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OrderBookLevel>()?;
    m.add_class::<OrderBookSnapshot>()?;
    m.add_class::<OrderBookUpdate>()?;
    m.add_class::<LOBAnalytics>()?;
    m.add_function(wrap_pyfunction!(calculate_lob_analytics, m)?)?;
    m.add_function(wrap_pyfunction!(apply_orderbook_update, m)?)?;
    m.add_function(wrap_pyfunction!(parse_binance_orderbook, m)?)?;
    Ok(())
}
