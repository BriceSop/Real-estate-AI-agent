from datetime import date
from dateutil.relativedelta import relativedelta

from app.data.repositories.transaction_repository import TransactionRepository

class EstimationService:
    """
    Service implementing real-estate price estimation.

    The service retrieves aggregated data through ``TransactionRepository`` and
    transforms it into structured results that can be exposed to an AI agent.
    """

    def __init__(self, transaction_repository: TransactionRepository) -> None:
        """
        Initialize the estimation service.

        Parameters
        ----------
        transaction_repository : TransactionRepository
            Repository providing access to aggregated transaction data.
        """
        self.repository = transaction_repository
        self.SURFACE_TOLERANCE = 0.20
        self.MINIMUM_COMPARABLES = 10

    def estimate_property(
        self,
        postal_code: str,
        property_type: str,
        reference_date: date,
        surface: float,
        rooms: int | None = None
    ) -> dict:
        """
        Return an estimation regarding price metrics for a given type of real-estate property.

        Parameters
        ----------
        postal_code : str
            Postal code identifying the requested area.
        property_type : str
            Principal property type.
        surface : float
            Surface in square meters.
        rooms : int
            Number of rooms available in the property.
        reference_date : date
            Date used as a reference for estimation.

        Returns
        -------
        dict[str, Any]
            Structured market metrics, or a ``not_found`` result when no
            matching data exists.
        """
        if surface <= 0:
            raise ValueError("La surface doit être strictement positive.")
        
        min_surface = surface * (1 - self.SURFACE_TOLERANCE)
        max_surface = surface * (1 + self.SURFACE_TOLERANCE)
        start_date = reference_date - relativedelta(months=3)

        metrics = self.repository.get_comparable_metrics(
            postal_code=postal_code,
            property_type=property_type,
            start_date=start_date,
            end_date=reference_date,
            min_surface=min_surface,
            max_surface=max_surface,
            rooms=rooms
        )

        if metrics is None:
            return {
                "status": "not_found",
                "message": "Aucune transaction comparable trouvée.",
            }

        return {
            "status": "success",
            "estimated_price": round(
                metrics.median_price_m2 * surface,
                -3,
            ),
            "low_estimate": round(
                metrics.q1_price_m2 * surface,
                -3,
            ),
            "high_estimate": round(
                metrics.q3_price_m2 * surface,
                -3,
            ),
            "median_price_m2": metrics.median_price_m2,
            "comparable_count": metrics.transaction_count,
            "insufficient_data": (
                metrics.transaction_count < self.MINIMUM_COMPARABLES
            ),
        }
