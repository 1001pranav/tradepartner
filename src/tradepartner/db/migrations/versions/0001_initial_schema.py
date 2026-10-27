"""initial schema

Revision ID: 0001
Revises: 
"""
import sqlalchemy as sa
from alembic import op

import tradepartner.db.models

# Frozen at this revision: later tables get their own triggers in later migrations.
APPEND_ONLY_TABLES = (
    "news_raw",
    "news_scores",
    "signals",
    "order_events",
    "fills",
    "charges",
    "cash_ledger",
    "risk_events",
    "audit_log",
)


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('audit_log',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('actor', sa.String(), nullable=False),
    sa.Column('action', sa.String(), nullable=False),
    sa.Column('payload_json', sa.JSON(), nullable=False),
    sa.Column('payload_hash', sa.String(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("actor IN ('user','system','scheduler')"),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_log_at'), 'audit_log', ['at'], unique=False)
    op.create_table('cash_ledger',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('date', sa.Date(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('amount_paise', sa.Integer(), nullable=False),
    sa.Column('ref', sa.String(), nullable=True),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("type IN ('funds_in','funds_out','charge','dividend','pnl_settlement','adjustment')"),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cash_ledger_date'), 'cash_ledger', ['date'], unique=False)
    op.create_table('corporate_actions',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('isin', sa.String(), nullable=True),
    sa.Column('action_type', sa.String(), nullable=False),
    sa.Column('ex_date', sa.Date(), nullable=False),
    sa.Column('ratio_num', sa.Integer(), nullable=True),
    sa.Column('ratio_den', sa.Integer(), nullable=True),
    sa.Column('amount_paise', sa.Integer(), nullable=True),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("action_type IN ('split','bonus','dividend','rights','merger')"),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_corporate_actions_symbol'), 'corporate_actions', ['symbol'], unique=False)
    op.create_table('feature_snapshots',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('as_of', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('feature_set_version', sa.String(), nullable=False),
    sa.Column('parquet_path', sa.String(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_feature_snapshots_as_of'), 'feature_snapshots', ['as_of'], unique=False)
    op.create_index(op.f('ix_feature_snapshots_symbol'), 'feature_snapshots', ['symbol'], unique=False)
    op.create_table('instruments',
    sa.Column('instrument_token', sa.Integer(), nullable=False),
    sa.Column('exchange', sa.String(), nullable=False),
    sa.Column('tradingsymbol', sa.String(), nullable=False),
    sa.Column('isin', sa.String(), nullable=True),
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('segment', sa.String(), nullable=False),
    sa.Column('lot_size', sa.Integer(), nullable=False),
    sa.Column('tick_size_paise', sa.Integer(), nullable=False),
    sa.Column('last_updated', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.PrimaryKeyConstraint('instrument_token')
    )
    op.create_index(op.f('ix_instruments_isin'), 'instruments', ['isin'], unique=False)
    op.create_index(op.f('ix_instruments_tradingsymbol'), 'instruments', ['tradingsymbol'], unique=False)
    op.create_table('job_runs',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('job', sa.String(), nullable=False),
    sa.Column('started_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('ended_at', tradepartner.db.models.UTCDateTime(), nullable=True),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_runs_job'), 'job_runs', ['job'], unique=False)
    op.create_table('mode_state',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('scope', sa.String(), nullable=False),
    sa.Column('mode', sa.String(), nullable=False),
    sa.Column('auto_unlocked', sa.Boolean(), nullable=False),
    sa.Column('changed_by', sa.String(), nullable=False),
    sa.Column('reason', sa.Text(), nullable=True),
    sa.Column('changed_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("changed_by IN ('user','system')"),
    sa.CheckConstraint("mode IN ('OFF','MANUAL','AUTO')"),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_mode_state_scope'), 'mode_state', ['scope'], unique=False)
    op.create_table('models',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('strategy', sa.String(), nullable=False),
    sa.Column('version', sa.String(), nullable=False),
    sa.Column('algo', sa.String(), nullable=False),
    sa.Column('feature_set_version', sa.String(), nullable=False),
    sa.Column('train_start', sa.Date(), nullable=False),
    sa.Column('train_end', sa.Date(), nullable=False),
    sa.Column('params_json', sa.JSON(), nullable=False),
    sa.Column('metrics_json', sa.JSON(), nullable=False),
    sa.Column('artifact_path', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('approved_by', sa.String(), nullable=True),
    sa.Column('approved_at', tradepartner.db.models.UTCDateTime(), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("status IN ('candidate','shadow','approved','retired')"),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('strategy', 'version')
    )
    op.create_index(op.f('ix_models_strategy'), 'models', ['strategy'], unique=False)
    op.create_table('news_raw',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('url', sa.String(), nullable=False),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('body', sa.Text(), nullable=True),
    sa.Column('published_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('published_at_estimated', sa.Boolean(), nullable=False),
    sa.Column('fetched_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('content_hash', sa.String(), nullable=False),
    sa.Column('lang', sa.String(), nullable=True),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('content_hash'),
    sa.UniqueConstraint('url')
    )
    op.create_table('positions_snapshot',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('snapshot_date', sa.Date(), nullable=False),
    sa.Column('kind', sa.String(), nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('isin', sa.String(), nullable=True),
    sa.Column('product', sa.String(), nullable=True),
    sa.Column('qty', sa.Integer(), nullable=False),
    sa.Column('avg_price_paise', sa.Integer(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("kind IN ('position','holding')"),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_positions_snapshot_snapshot_date'), 'positions_snapshot', ['snapshot_date'], unique=False)
    op.create_table('risk_events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('rule', sa.String(), nullable=False),
    sa.Column('severity', sa.String(), nullable=False),
    sa.Column('scope', sa.String(), nullable=False),
    sa.Column('details_json', sa.JSON(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("severity IN ('info','block','halt')"),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_risk_events_at'), 'risk_events', ['at'], unique=False)
    op.create_table('trading_calendar',
    sa.Column('date', sa.Date(), nullable=False),
    sa.Column('is_trading_day', sa.Boolean(), nullable=False),
    sa.Column('session_open', sa.String(), nullable=True),
    sa.Column('session_close', sa.String(), nullable=True),
    sa.Column('note', sa.String(), nullable=True),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.PrimaryKeyConstraint('date')
    )
    op.create_table('universe_members',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('universe_name', sa.String(), nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('valid_from', sa.Date(), nullable=False),
    sa.Column('valid_to', sa.Date(), nullable=True),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_universe_members_symbol'), 'universe_members', ['symbol'], unique=False)
    op.create_index(op.f('ix_universe_members_universe_name'), 'universe_members', ['universe_name'], unique=False)
    op.create_table('news_entities',
    sa.Column('news_id', sa.Integer(), nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('match_method', sa.String(), nullable=False),
    sa.Column('confidence', sa.Double(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("match_method IN ('alias','isin','llm')"),
    sa.ForeignKeyConstraint(['news_id'], ['news_raw.id'], ),
    sa.PrimaryKeyConstraint('news_id', 'symbol')
    )
    op.create_table('news_scores',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('news_id', sa.Integer(), nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('llm_model', sa.String(), nullable=False),
    sa.Column('llm_model_digest', sa.String(), nullable=True),
    sa.Column('prompt_version', sa.String(), nullable=False),
    sa.Column('relevance', sa.Double(), nullable=False),
    sa.Column('sentiment', sa.Double(), nullable=False),
    sa.Column('event_type', sa.String(), nullable=False),
    sa.Column('materiality', sa.Double(), nullable=False),
    sa.Column('horizon', sa.String(), nullable=False),
    sa.Column('confidence', sa.Double(), nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('raw_json', sa.JSON(), nullable=False),
    sa.Column('scored_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("horizon IN ('intraday','days','weeks','months')"),
    sa.CheckConstraint('confidence BETWEEN 0 AND 1'),
    sa.CheckConstraint('materiality BETWEEN 0 AND 1'),
    sa.CheckConstraint('relevance BETWEEN 0 AND 1'),
    sa.CheckConstraint('sentiment BETWEEN -1 AND 1'),
    sa.ForeignKeyConstraint(['news_id'], ['news_raw.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_news_scores_news_id'), 'news_scores', ['news_id'], unique=False)
    op.create_index(op.f('ix_news_scores_symbol'), 'news_scores', ['symbol'], unique=False)
    op.create_table('signals',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('strategy', sa.String(), nullable=False),
    sa.Column('model_id', sa.Integer(), nullable=True),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('side', sa.String(), nullable=False),
    sa.Column('as_of', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('score', sa.Double(), nullable=False),
    sa.Column('prob', sa.Double(), nullable=True),
    sa.Column('expected_return', sa.Double(), nullable=True),
    sa.Column('horizon_days', sa.Integer(), nullable=False),
    sa.Column('explanation', sa.Text(), nullable=True),
    sa.Column('features_ref', sa.String(), nullable=True),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("side IN ('BUY','SELL')"),
    sa.CheckConstraint("status IN ('new','proposed','rejected_by_risk','shown','approved','executed','expired','ignored')"),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_signals_as_of'), 'signals', ['as_of'], unique=False)
    op.create_index(op.f('ix_signals_strategy'), 'signals', ['strategy'], unique=False)
    op.create_index(op.f('ix_signals_symbol'), 'signals', ['symbol'], unique=False)
    op.create_table('order_proposals',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('signal_id', sa.Integer(), nullable=True),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('side', sa.String(), nullable=False),
    sa.Column('product', sa.String(), nullable=False),
    sa.Column('order_type', sa.String(), nullable=False),
    sa.Column('qty', sa.Integer(), nullable=False),
    sa.Column('limit_price_paise', sa.Integer(), nullable=True),
    sa.Column('trigger_price_paise', sa.Integer(), nullable=True),
    sa.Column('stop_loss_paise', sa.Integer(), nullable=True),
    sa.Column('target_paise', sa.Integer(), nullable=True),
    sa.Column('time_stop_date', sa.Date(), nullable=True),
    sa.Column('risk_amount_paise', sa.Integer(), nullable=True),
    sa.Column('mode_at_creation', sa.String(), nullable=False),
    sa.Column('risk_decision', sa.String(), nullable=True),
    sa.Column('risk_reasons_json', sa.JSON(), nullable=True),
    sa.Column('idempotency_key', sa.String(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("mode_at_creation IN ('OFF','MANUAL','AUTO')"),
    sa.CheckConstraint("order_type IN ('LIMIT','SL','SL-M','MARKET')"),
    sa.CheckConstraint("product IN ('CNC','MIS')"),
    sa.CheckConstraint("risk_decision IS NULL OR risk_decision IN ('approved','rejected')"),
    sa.CheckConstraint("side IN ('BUY','SELL')"),
    sa.ForeignKeyConstraint(['signal_id'], ['signals.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('idempotency_key')
    )
    op.create_index(op.f('ix_order_proposals_symbol'), 'order_proposals', ['symbol'], unique=False)
    op.create_table('orders',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('proposal_id', sa.Integer(), nullable=True),
    sa.Column('broker_order_id', sa.String(), nullable=True),
    sa.Column('origin', sa.String(), nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('side', sa.String(), nullable=False),
    sa.Column('product', sa.String(), nullable=False),
    sa.Column('order_type', sa.String(), nullable=False),
    sa.Column('qty', sa.Integer(), nullable=False),
    sa.Column('price_paise', sa.Integer(), nullable=True),
    sa.Column('trigger_paise', sa.Integer(), nullable=True),
    sa.Column('tag', sa.String(length=20), nullable=True),
    sa.Column('placed_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("origin IN ('bot_auto','bot_manual_api','manual_external','paper')"),
    sa.CheckConstraint("side IN ('BUY','SELL')"),
    sa.ForeignKeyConstraint(['proposal_id'], ['order_proposals.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('broker_order_id')
    )
    op.create_index(op.f('ix_orders_symbol'), 'orders', ['symbol'], unique=False)
    op.create_table('fills',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('broker_trade_id', sa.String(), nullable=False),
    sa.Column('broker_order_id', sa.String(), nullable=True),
    sa.Column('order_id', sa.Integer(), nullable=True),
    sa.Column('origin', sa.String(), nullable=False),
    sa.Column('account', sa.String(), nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('isin', sa.String(), nullable=True),
    sa.Column('exchange', sa.String(), nullable=False),
    sa.Column('side', sa.String(), nullable=False),
    sa.Column('product', sa.String(), nullable=False),
    sa.Column('qty', sa.Integer(), nullable=False),
    sa.Column('price_paise', sa.Integer(), nullable=False),
    sa.Column('filled_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('trade_date', sa.Date(), nullable=False),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("exchange = 'NSE'"),
    sa.CheckConstraint("origin IN ('bot_auto','bot_manual_api','manual_external','paper')"),
    sa.CheckConstraint("side IN ('BUY','SELL')"),
    sa.CheckConstraint("source IN ('kite_api','console_csv','paper','correction')"),
    sa.CheckConstraint('price_paise > 0'),
    sa.CheckConstraint('qty > 0'),
    sa.ForeignKeyConstraint(['order_id'], ['orders.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('broker_trade_id')
    )
    op.create_index(op.f('ix_fills_isin'), 'fills', ['isin'], unique=False)
    op.create_index(op.f('ix_fills_symbol'), 'fills', ['symbol'], unique=False)
    op.create_index(op.f('ix_fills_trade_date'), 'fills', ['trade_date'], unique=False)
    op.create_table('order_events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('order_id', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('filled_qty', sa.Integer(), nullable=False),
    sa.Column('avg_price_paise', sa.Integer(), nullable=True),
    sa.Column('message', sa.Text(), nullable=True),
    sa.Column('at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['order_id'], ['orders.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_order_events_order_id'), 'order_events', ['order_id'], unique=False)
    op.create_table('charges',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('fill_id', sa.Integer(), nullable=True),
    sa.Column('trade_date', sa.Date(), nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('component', sa.String(), nullable=False),
    sa.Column('amount_paise', sa.Integer(), nullable=False),
    sa.Column('computed', sa.Boolean(), nullable=False),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("component IN ('brokerage','stt','exchange_txn','sebi','stamp','gst','dp','other')"),
    sa.CheckConstraint("source IN ('computed','contract_note')"),
    sa.ForeignKeyConstraint(['fill_id'], ['fills.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_charges_trade_date'), 'charges', ['trade_date'], unique=False)
    op.create_table('lots',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('isin', sa.String(), nullable=True),
    sa.Column('open_fill_id', sa.Integer(), nullable=False),
    sa.Column('open_date', sa.Date(), nullable=False),
    sa.Column('qty_open', sa.Integer(), nullable=False),
    sa.Column('qty_remaining', sa.Integer(), nullable=False),
    sa.Column('cost_paise_per_share', sa.Integer(), nullable=False),
    sa.Column('adjusted_for_ca', sa.Boolean(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.ForeignKeyConstraint(['open_fill_id'], ['fills.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_lots_symbol'), 'lots', ['symbol'], unique=False)
    op.create_table('realized',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('symbol', sa.String(), nullable=False),
    sa.Column('open_fill_id', sa.Integer(), nullable=True),
    sa.Column('close_fill_id', sa.Integer(), nullable=False),
    sa.Column('qty', sa.Integer(), nullable=False),
    sa.Column('buy_value_paise', sa.Integer(), nullable=False),
    sa.Column('sell_value_paise', sa.Integer(), nullable=False),
    sa.Column('charges_paise', sa.Integer(), nullable=False),
    sa.Column('stt_paise', sa.Integer(), nullable=False),
    sa.Column('holding_days', sa.Integer(), nullable=False),
    sa.Column('category', sa.String(), nullable=False),
    sa.Column('fy', sa.String(), nullable=False),
    sa.Column('pnl_paise', sa.Integer(), nullable=False),
    sa.Column('created_at', tradepartner.db.models.UTCDateTime(), nullable=False),
    sa.CheckConstraint("category IN ('intraday_spec','stcg','ltcg','business','unmatched')"),
    sa.ForeignKeyConstraint(['close_fill_id'], ['fills.id'], ),
    sa.ForeignKeyConstraint(['open_fill_id'], ['fills.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_realized_fy'), 'realized', ['fy'], unique=False)
    op.create_index(op.f('ix_realized_symbol'), 'realized', ['symbol'], unique=False)

    # CLAUDE.md rule 7: append-only tables reject UPDATE and DELETE. Corrections are new rows.
    for table in APPEND_ONLY_TABLES:
        for verb in ("UPDATE", "DELETE"):
            op.execute(
                f"CREATE TRIGGER {table}_no_{verb.lower()} BEFORE {verb} ON {table} "
                f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"
            )


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_realized_symbol'), table_name='realized')
    op.drop_index(op.f('ix_realized_fy'), table_name='realized')
    op.drop_table('realized')
    op.drop_index(op.f('ix_lots_symbol'), table_name='lots')
    op.drop_table('lots')
    op.drop_index(op.f('ix_charges_trade_date'), table_name='charges')
    op.drop_table('charges')
    op.drop_index(op.f('ix_order_events_order_id'), table_name='order_events')
    op.drop_table('order_events')
    op.drop_index(op.f('ix_fills_trade_date'), table_name='fills')
    op.drop_index(op.f('ix_fills_symbol'), table_name='fills')
    op.drop_index(op.f('ix_fills_isin'), table_name='fills')
    op.drop_table('fills')
    op.drop_index(op.f('ix_orders_symbol'), table_name='orders')
    op.drop_table('orders')
    op.drop_index(op.f('ix_order_proposals_symbol'), table_name='order_proposals')
    op.drop_table('order_proposals')
    op.drop_index(op.f('ix_signals_symbol'), table_name='signals')
    op.drop_index(op.f('ix_signals_strategy'), table_name='signals')
    op.drop_index(op.f('ix_signals_as_of'), table_name='signals')
    op.drop_table('signals')
    op.drop_index(op.f('ix_news_scores_symbol'), table_name='news_scores')
    op.drop_index(op.f('ix_news_scores_news_id'), table_name='news_scores')
    op.drop_table('news_scores')
    op.drop_table('news_entities')
    op.drop_index(op.f('ix_universe_members_universe_name'), table_name='universe_members')
    op.drop_index(op.f('ix_universe_members_symbol'), table_name='universe_members')
    op.drop_table('universe_members')
    op.drop_table('trading_calendar')
    op.drop_index(op.f('ix_risk_events_at'), table_name='risk_events')
    op.drop_table('risk_events')
    op.drop_index(op.f('ix_positions_snapshot_snapshot_date'), table_name='positions_snapshot')
    op.drop_table('positions_snapshot')
    op.drop_table('news_raw')
    op.drop_index(op.f('ix_models_strategy'), table_name='models')
    op.drop_table('models')
    op.drop_index(op.f('ix_mode_state_scope'), table_name='mode_state')
    op.drop_table('mode_state')
    op.drop_index(op.f('ix_job_runs_job'), table_name='job_runs')
    op.drop_table('job_runs')
    op.drop_index(op.f('ix_instruments_tradingsymbol'), table_name='instruments')
    op.drop_index(op.f('ix_instruments_isin'), table_name='instruments')
    op.drop_table('instruments')
    op.drop_index(op.f('ix_feature_snapshots_symbol'), table_name='feature_snapshots')
    op.drop_index(op.f('ix_feature_snapshots_as_of'), table_name='feature_snapshots')
    op.drop_table('feature_snapshots')
    op.drop_index(op.f('ix_corporate_actions_symbol'), table_name='corporate_actions')
    op.drop_table('corporate_actions')
    op.drop_index(op.f('ix_cash_ledger_date'), table_name='cash_ledger')
    op.drop_table('cash_ledger')
    op.drop_index(op.f('ix_audit_log_at'), table_name='audit_log')
    op.drop_table('audit_log')
